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

// Add vault: any signed-in user can create one and becomes its Owner.
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['vaultName'])) {
    $conn->begin_transaction();

    $stmt = $conn->prepare("INSERT INTO vaults (vault_name) VALUES (?)");
    $stmt->bind_param("s", $_POST['vaultName']);
    $stmt->execute();
    $newVaultId = $conn->insert_id;

    $stmt = $conn->prepare(
        "INSERT INTO vault_permissions (user_id, vault_id, role_id)
         SELECT user_id, ?, 1 FROM users WHERE username = ?"
    );
    $stmt->bind_param("is", $newVaultId, $_SESSION['authenticated']);
    $stmt->execute();

    if ($stmt->affected_rows !== 1) {
        $conn->rollback();
        die('A fatal error occurred while adding the vault.');
    }
    $conn->commit();
    header("Location: index.php");
    exit();
}

// Rename vault (Owner only)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['editVaultName'], $_POST['editVaultId'])) {
    $vaultId = (int)$_POST['editVaultId'];
    requireVaultRole($conn, $vaultId, 'Owner');

    $stmt = $conn->prepare("UPDATE vaults SET vault_name = ? WHERE vault_id = ?");
    $stmt->bind_param("si", $_POST['editVaultName'], $vaultId);
    $stmt->execute();
    header("Location: index.php");
    exit();
}

// Delete vault (Owner only)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && !empty($_POST['deleteVaultId'])) {
    $vaultId = (int)$_POST['deleteVaultId'];
    requireVaultRole($conn, $vaultId, 'Owner');

    $stmt = $conn->prepare("DELETE FROM vaults WHERE vault_id = ?");
    $stmt->bind_param("i", $vaultId);
    $stmt->execute();
    header("Location: index.php");
    exit();
}

// List the vaults this user can see (all of them for a site admin), optionally filtered by search.
$searchQuery = trim($_GET['searchQuery'] ?? '');
$like = '%' . $searchQuery . '%';

if (!empty($_SESSION['isSiteAdministrator'])) {
    $stmt = $conn->prepare("SELECT vault_id, vault_name FROM vaults WHERE vault_name LIKE ? ORDER BY vault_id");
    $stmt->bind_param("s", $like);
} else {
    $stmt = $conn->prepare(
        "SELECT DISTINCT v.vault_id, v.vault_name
           FROM vaults v
           JOIN vault_permissions vp ON vp.vault_id = v.vault_id
           JOIN users u ON u.user_id = vp.user_id
          WHERE u.username = ? AND v.vault_name LIKE ?
          ORDER BY v.vault_id"
    );
    $stmt->bind_param("ss", $_SESSION['authenticated'], $like);
}
$stmt->execute();
$result = $stmt->get_result();
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Vaults</title>
    <!-- Add Bootstrap CSS link here -->
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
</head>

<body>

    <?php include '../components/nav-bar.php'; ?>

    <div class="container mt-4">
        <h2>Password Vaults</h2>

        <!-- Add button to open a modal for adding a new vault -->
        <button type="button" class="btn btn-primary mb-2" data-toggle="modal" data-target="#addVaultModal">
            Add Vault
        </button>

        <!-- Table to display vaults -->
        <table class="table">
            <thead>
                
                <input type="text" id="searchInput" onkeypress="searchTable(event)" placeholder="Search for vaults..."
                    class="form-control mb-3">
                    <?php if (!empty ($searchQuery)) {
                    // If $searchQuery is not blank, display the label with its value
                    echo "<label>Search Results for : " . htmlspecialchars($searchQuery, ENT_QUOTES) . "</label>"; 
                } ?>
                    <table class="table table-bordered" id="vaultTable">
                    <thead>
                        <tr>
                            <th>Vault Name</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                    <tbody>

                        <?php while ($row = $result->fetch_assoc()): ?>
                            <tr>
                                <td>
                                    <?php echo htmlspecialchars((string)$row['vault_name'], ENT_QUOTES); ?>
                                </td>
                                <td>
                                    <a href="vault_details.php?vault_id=<?php echo htmlspecialchars((string)$row['vault_id'], ENT_QUOTES); ?>"
                                        class="btn btn-primary btn-sm" role="button" aria-disabled="true">View Vault</a>

                                    <!-- Edit button to open a modal for editing a vault -->
                                    <button class="btn btn-warning btn-sm edit-btn" data-toggle="modal"
                                        data-target="#editVaultModal" data-vault-name="<?php echo htmlspecialchars((string)$row['vault_name'], ENT_QUOTES); ?>"
                                        data-vault-id="<?php echo htmlspecialchars((string)$row['vault_id'], ENT_QUOTES); ?>">Edit</button>

                                    <!-- Delete button to open a modal for deleting a vault -->
                                    <button class="btn btn-danger btn-sm delete-btn" data-toggle="modal"
                                        data-target="#deleteVaultModal" data-vault-name="<?php echo htmlspecialchars((string)$row['vault_name'], ENT_QUOTES); ?>"
                                        data-vault-id="<?php echo htmlspecialchars((string)$row['vault_id'], ENT_QUOTES); ?>">Delete</button>
                                </td>
                            </tr>
                        <?php endwhile; ?>
                    </tbody>
                </table>
    </div>

    <!-- Modal for adding a new vault -->
    <div class="modal" id="addVaultModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Add New Vault</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <!-- Add form for adding a new vault here -->
                    <form method="POST" id="addVaultForm">
                        <div class="form-group">
                            <label for="vaultName">Vault Name:</label>
                            <input type="text" class="form-control" id="vaultName" name="vaultName" required>
                        </div>
                        <button type="submit" class="btn btn-primary">Add Vault</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Modal for editing a vault -->
    <div class="modal" id="editVaultModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Edit Vault</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <!-- Add form for editing a vault here -->
                    <form method="POST" id="editVaultForm">
                        <div class="form-group">
                            <input type="hidden" id="editVaultId" name="editVaultId">
                            <label for="editVaultName">Vault Name:</label>
                            <input type="text" class="form-control" id="editVaultName" name="editVaultName" required>
                        </div>
                        <button type="submit" class="btn btn-warning">Update Vault</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Modal for deleting a vault -->
    <div class="modal" id="deleteVaultModal">
        <div class="modal-dialog">
            <div class="modal-content">

                <!-- Modal Header -->
                <div class="modal-header">
                    <h4 class="modal-title">Delete Vault</h4>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>

                <!-- Modal Body -->
                <div class="modal-body">
                    <p id="deleteWarningPara"></p>
                    <!-- Add hidden input for vault ID -->
                    <form method="POST" id="deleteVaultForm">
                        <input type="hidden" id="deleteVaultId" name="deleteVaultId">
                        <button type="submit" class="btn btn-danger" id="confirmDeleteBtn">Delete</button>
                    </form>
                </div>

            </div>
        </div>
    </div>

    <!-- Add Bootstrap JS and Popper.js scripts here -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
    <!-- Add your custom JavaScript script for handling modals, filtering, and row click redirection -->
    <script>


        function searchTable(event) {

            if (event.keyCode === 13) {
                var searchInput = document.getElementById("searchInput").value;
                window.location.href = "./index.php?searchQuery=" + searchInput;
            }


            // filter = input.value.toUpperCase();
            // table = document.getElementById("vaultTable");
            // tr = table.getElementsByTagName("tr");

            // for (i = 0; i < tr.length; i++) {
            //     td = tr[i].getElementsByTagName("td")[0]; // Change index based on the column you want to search
            //     if (td) {
            //         txtValue = td.textContent || td.innerText;
            //         if (txtValue.toUpperCase().indexOf(filter) > -1) {
            //             tr[i].style.display = "";
            //         } else {
            //             tr[i].style.display = "none";
            //         }
            //     }
            // }        
        }


        // Handle edit button click
        var editButtons = document.querySelectorAll('.edit-btn');
        editButtons.forEach(function (button) {
            button.addEventListener('click', function () {
                var vaultId = button.getAttribute('data-vault-id');
                var vaultName = button.getAttribute('data-vault-name');
                // You can use vaultId to populate the edit modal, if needed
                document.getElementById('editVaultId').value = vaultId;
                document.getElementById('editVaultName').value = vaultName;
            });
        });

        // Handle delete button click
        var deleteButtons = document.querySelectorAll('.delete-btn');
        deleteButtons.forEach(function (button) {
            button.addEventListener('click', function () {
                var vaultId = button.getAttribute('data-vault-id');
                var vaultName = button.getAttribute('data-vault-name');
                // You can use vaultId to populate the delete modal, if needed

                document.getElementById('deleteVaultId').value = vaultId;
                document.getElementById('deleteWarningPara').innerText = 'Are you sure you want to delete the ' + vaultName + ' vault?';
            });
        });


    </script>
</body>

</html>

<?php
$conn->close();
?>