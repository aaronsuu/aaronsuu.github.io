<?php
include __DIR__ . '/../cwechecks.php';
include __DIR__ . '/../components/loggly-logger.php';
require_once __DIR__ . '/../components/vault-access.php';

$hostname = 'backend-mysql-database';
$username = 'user';
$password = getenv('MYSQL_PASSWORD');
$database = 'password_manager';

$conn = new mysqli($hostname, $username, $password, $database);
if ($conn->connect_error) {
    die('A fatal error occurred and has been logged.');
}

$uploadDir = './uploads/';

function redirectToVault(int $vaultId): void
{
    header("Location: vault_details.php?vault_id=$vaultId");
    exit();
}

// ---- Write actions. Each one checks the user's role on the vault before touching it. ----

// Add password (Editor or Owner)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['addUsername'], $_POST['addWebsite'], $_POST['addPassword'], $_POST['vaultId'])) {
    $vaultId = (int)$_POST['vaultId'];
    requireVaultRole($conn, $vaultId, 'Editor');

    $filePath = storeUpload($_FILES['file'] ?? [], $uploadDir);
    $notes = $_POST['addNotes'] ?? '';

    $stmt = $conn->prepare(
        "INSERT INTO vault_passwords (vault_id, username, website, password, notes, file_path) VALUES (?, ?, ?, ?, ?, ?)"
    );
    $stmt->bind_param("isssss", $vaultId, $_POST['addUsername'], $_POST['addWebsite'], $_POST['addPassword'], $notes, $filePath);
    if (!$stmt->execute()) {
        die('A fatal error occurred and has been logged.');
    }
    redirectToVault($vaultId);
}

// Edit password (Editor or Owner, and the row must belong to this vault)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['editPasswordId'], $_POST['editUsername'], $_POST['editPassword'], $_POST['editWebsite'], $_POST['vaultId'])) {
    $vaultId = (int)$_POST['vaultId'];
    $passwordId = (int)$_POST['editPasswordId'];
    requireVaultRole($conn, $vaultId, 'Editor');
    $existing = requirePasswordInVault($conn, $passwordId, $vaultId);

    $filePath = storeUpload($_FILES['editFile'] ?? [], $uploadDir) ?? $existing['file_path'];
    $notes = $_POST['editNotes'] ?? '';

    $stmt = $conn->prepare(
        "UPDATE vault_passwords SET username = ?, website = ?, password = ?, notes = ?, file_path = ?
          WHERE password_id = ? AND vault_id = ?"
    );
    $stmt->bind_param("sssssii", $_POST['editUsername'], $_POST['editWebsite'], $_POST['editPassword'], $notes, $filePath, $passwordId, $vaultId);
    if (!$stmt->execute()) {
        die('A fatal error occurred and has been logged.');
    }
    redirectToVault($vaultId);
}

// Delete password (Owner only)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['deletePasswordId'], $_POST['vaultId'])) {
    $vaultId = (int)$_POST['vaultId'];
    $passwordId = (int)$_POST['deletePasswordId'];
    requireVaultRole($conn, $vaultId, 'Owner');
    requirePasswordInVault($conn, $passwordId, $vaultId);

    $stmt = $conn->prepare("DELETE FROM vault_passwords WHERE password_id = ? AND vault_id = ?");
    $stmt->bind_param("ii", $passwordId, $vaultId);
    if (!$stmt->execute()) {
        die('A fatal error occurred and has been logged.');
    }
    redirectToVault($vaultId);
}

// Delete an attachment (Editor or Owner)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['deleteFilePasswordId'], $_POST['deleteFileVaultId'], $_POST['deleteFileSubmit'])) {
    $vaultId = (int)$_POST['deleteFileVaultId'];
    $passwordId = (int)$_POST['deleteFilePasswordId'];
    requireVaultRole($conn, $vaultId, 'Editor');
    $row = requirePasswordInVault($conn, $passwordId, $vaultId);

    // Only ever unlink files inside the uploads directory.
    $target = $row['file_path'] ? realpath($row['file_path']) : false;
    $uploadsRoot = realpath($uploadDir);
    if ($target && $uploadsRoot && strpos($target, $uploadsRoot . DIRECTORY_SEPARATOR) === 0) {
        unlink($target);
    }

    $stmt = $conn->prepare("UPDATE vault_passwords SET file_path = NULL WHERE password_id = ? AND vault_id = ?");
    $stmt->bind_param("ii", $passwordId, $vaultId);
    $stmt->execute();
    redirectToVault($vaultId);
}

// ---- Read: the vault page itself (any role on the vault) ----

$vaultId = isset($_GET['vault_id']) ? (int)$_GET['vault_id'] : 0;
if ($vaultId <= 0) {
    die('Invalid vault.');
}
$role = requireVaultRole($conn, $vaultId, 'Viewer');
$isVaultOwner = ($role === 'Owner');

$stmt = $conn->prepare("SELECT vault_name FROM vaults WHERE vault_id = ?");
$stmt->bind_param("i", $vaultId);
$stmt->execute();
$vaultName = $stmt->get_result()->fetch_assoc()['vault_name'] ?? '';
$stmt->close();

$searchQuery = trim($_GET['searchQuery'] ?? '');
if ($searchQuery !== '') {
    $like = '%' . $searchQuery . '%';
    $stmt = $conn->prepare(
        "SELECT * FROM vault_passwords WHERE vault_id = ? AND (username LIKE ? OR website LIKE ?)"
    );
    $stmt->bind_param("iss", $vaultId, $like, $like);
} else {
    $stmt = $conn->prepare("SELECT * FROM vault_passwords WHERE vault_id = ?");
    $stmt->bind_param("i", $vaultId);
}
$stmt->execute();
$resultPasswords = $stmt->get_result();
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>
        <?php echo htmlspecialchars((string)$vaultName, ENT_QUOTES); ?> Vault
    </title>
    <!-- Add Bootstrap CSS link here -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
</head>

<body>

    <?php include '../components/nav-bar.php'; ?>


    <div class="container mt-4">
        <h2>
            <?php echo htmlspecialchars((string)$vaultName, ENT_QUOTES); ?> Vault Passwords
        </h2>
        <button type="button" class="btn btn-primary mb-2" data-toggle="modal" data-target="#addPasswordModal">
            Add Password
        </button>
        <?php if ($isVaultOwner): ?>
            <a href="./vault_permissions.php?vault_id=<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>" class="btn btn-warning mb-2"> Edit Vault
                Permissons </a>
        <?php endif; ?>

        <input type="text" id="searchInput" onkeyup="searchTable(event)" placeholder="Search for passwords..."
            class="form-control mb-3">
            <?php if (!empty ($searchQuery)) {
                    // If $searchQuery is not blank, display the label with its value
                    echo "<label>Search Results for : " . htmlspecialchars($searchQuery, ENT_QUOTES) . "</label>"; 
                } ?>
        <table class="table table-bordered" id="passwordTable">
            <thead>
                <tr>
                    <th>Username</th>
                    <th>Website</th>
                    <th>Password</th>
                    <th>Notes</th>
                    <th>Actions</th>
                    <th>File</th>
                </tr>
            </thead>
            <tbody>
                <?php while ($rowPassword = $resultPasswords->fetch_assoc()): ?>
                    <tr data-password-id="<?php echo htmlspecialchars((string)$rowPassword['password_id'], ENT_QUOTES); ?>">
                        <td>
                            <?php echo htmlspecialchars((string)$rowPassword['username'], ENT_QUOTES); ?>
                        </td>
                        <td>
                            <?php echo htmlspecialchars((string)$rowPassword['website'], ENT_QUOTES); ?>
                        </td>
                        <td>
                            <input type="password" class="password-field" value="<?php echo htmlspecialchars((string)$rowPassword['password'], ENT_QUOTES); ?>"
                                disabled>
                        </td>
                        <td>
                            <?php echo htmlspecialchars((string)$rowPassword['notes'], ENT_QUOTES); ?>
                        </td>
                        <td>
                            <button class="btn btn-primary btn-sm show-password-btn">Show Password</button>
                            <button class="btn btn-warning btn-sm edit-password-btn" data-toggle="modal"
                                data-target="#editPasswordModal" data-password-notes="<?php echo htmlspecialchars((string)$rowPassword['notes'], ENT_QUOTES); ?>"
                                data-password-password="<?php echo htmlspecialchars((string)$rowPassword['password'], ENT_QUOTES); ?>"
                                data-password-website="<?php echo htmlspecialchars((string)$rowPassword['website'], ENT_QUOTES); ?>"
                                data-password-username="<?php echo htmlspecialchars((string)$rowPassword['username'], ENT_QUOTES); ?>"
                                data-password-id="<?php echo htmlspecialchars((string)$rowPassword['password_id'], ENT_QUOTES); ?>">Edit</button>
                            <button class="btn btn-danger btn-sm delete-password-btn" data-toggle="modal"
                                data-target="#deletePasswordModal"
                                data-password-id="<?php echo htmlspecialchars((string)$rowPassword['password_id'], ENT_QUOTES); ?>">Delete</button>
                        </td>
                        <td>
                            <?php if (!empty ($rowPassword['file_path'])): ?>
                                <a href="download_file.php?file=<?php echo urlencode($rowPassword['file_path']); ?>&vault_id=<?php echo urlencode($vaultId); ?>"
                                    target="_blank">Download File</a>
                                <form method="POST" action="<?php echo htmlspecialchars((string)$_SERVER['PHP_SELF'], ENT_QUOTES); ?>">
                                    <input type="hidden" name="deleteFilePasswordId"
                                        value="<?php echo htmlspecialchars((string)$rowPassword['password_id'], ENT_QUOTES); ?>">
                                    <input type="hidden" name="deleteFileVaultId" value="<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>">
                                    <button type="submit" name="deleteFileSubmit" class="btn btn-danger btn-sm">Delete
                                        File</button>
                                </form>
                            <?php endif; ?>
                        </td>
                    </tr>
                <?php endwhile; ?>
            </tbody>
        </table>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {
            var showPasswordButtons = document.querySelectorAll('.show-password-btn');
            showPasswordButtons.forEach(function (button) {
                button.addEventListener('click', function () {
                    var passwordField = button.closest('tr').querySelector('.password-field');
                    passwordField.type = (passwordField.type === 'password') ? 'text' : 'password';
                    if (button.textContent == 'Show Password') {
                        button.textContent = 'Hide Password';
                    } else {
                        button.textContent = 'Show Password';
                    }
                });
            });
        });
    </script>


    <div class="modal" id="addPasswordModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Add New Password</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <!-- Add form for adding a new password here -->
                    <form method="POST" id="addPasswordForm" enctype="multipart/form-data">
                        <input type="hidden" id="addVaultId" name="vaultId" value="<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>">
                        <div class="form-group">
                            <label for="addUsername">Username:</label>
                            <input type="text" class="form-control" id="addUsername" name="addUsername" required>
                        </div>
                        <div class="form-group">
                            <label for="addWebsite">Website:</label>
                            <input type="text" class="form-control" id="addWebsite" name="addWebsite" required>
                        </div>
                        <div class="form-group">
                            <label for="addPassword">Password:</label>
                            <input type="password" class="form-control" id="addPassword" name="addPassword" required>
                        </div>
                        <div class="form-group">
                            <label for="addNotes">Notes:</label>
                            <textarea class="form-control" id="addNotes" name="addNotes" rows="3"></textarea>
                        </div>
                        <div class="form-group">
                            <label for="file">File:</label>
                            <input type="file" name="file">
                        </div>
                        <button type="submit" class="btn btn-primary">Add Password</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Modal for editing a password -->
    <div class="modal" id="editPasswordModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Edit Password</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <!-- Add form for editing a password here -->
                    <form method="POST" id="editPasswordForm" enctype="multipart/form-data">
                        <input type="hidden" id="editVaultId" name="vaultId" value="<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>">
                        <div class="form-group">
                            <label for="editUsername">Username:</label>
                            <input type="text" class="form-control" id="editUsername" name="editUsername" required>
                        </div>
                        <div class="form-group">
                            <label for="editWebsite">Website:</label>
                            <input type="text" class="form-control" id="editWebsite" name="editWebsite" required>
                        </div>
                        <div class="form-group">
                            <label for="editPassword">Password:</label>
                            <input type="password" class="form-control" id="editPassword" name="editPassword" required>
                        </div>
                        <div class="form-group">
                            <label for="editNotes">Notes:</label>
                            <textarea class="form-control" id="editNotes" name="editNotes" rows="3"></textarea>
                        </div>
                        <div class="form-group">
                            <label for="editFile">File:</label>
                            <input type="file" name="editFile">
                        </div>
                        <input type="hidden" id="editPasswordId" name="editPasswordId">
                        <button type="submit" class="btn btn-warning">Update Password</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Modal for deleting a password -->
    <div class="modal" id="deletePasswordModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Delete Password</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <p>Are you sure you want to delete this password?</p>
                    <!-- Add hidden input for password ID -->
                    <form method="POST" id="deletePasswordForm">
                        <input type="hidden" id="deleteVaultId" name="vaultId" value="<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>">
                        <input type="hidden" id="deletePasswordId" name="deletePasswordId">
                        <button type="submit" class="btn btn-danger" id="confirmDeletePasswordBtn">Delete</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Add Bootstrap JS and Popper.js scripts here -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
    <!-- Add your custom JavaScript script for handling modals and row click redirection -->
    <script>

        function searchTable(event) {

            if (event.keyCode === 13) {
                var searchInput = document.getElementById("searchInput").value;
                window.location.href = "./vault_details.php?vault_id=<?php echo htmlspecialchars((string)$vaultId, ENT_QUOTES); ?>&searchQuery=" + searchInput;
            }
        }


        document.addEventListener("DOMContentLoaded", function () {
            // Handle edit button click for passwords
            var editPasswordButtons = document.querySelectorAll('.edit-password-btn');
            editPasswordButtons.forEach(function (button) {
                button.addEventListener('click', function () {

                    document.getElementById('editPasswordId').value = button.getAttribute('data-password-id');;
                    document.getElementById('editUsername').value = button.getAttribute('data-password-username');;
                    document.getElementById('editWebsite').value = button.getAttribute('data-password-website');;
                    document.getElementById('editPassword').value = button.getAttribute('data-password-password');;
                    document.getElementById('editNotes').value = button.getAttribute('data-password-notes');;
                });
            });

            // Handle delete button click for passwords
            var deletePasswordButtons = document.querySelectorAll('.delete-password-btn');
            deletePasswordButtons.forEach(function (button) {
                button.addEventListener('click', function () {
                    var passwordId = button.getAttribute('data-password-id');
                    console.log('Setting Delete Password ID to : ' + passwordId);
                    document.getElementById('deletePasswordId').value = passwordId
                });
            });
        });
    </script>
</body>

</html>

<?php
$conn->close();
?>