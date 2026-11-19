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

// Only a vault's Owner (or a site admin) can see or change who has access to it.
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action'], $_POST['vault_id'])) {
    $vaultId = (int)$_POST['vault_id'];
    requireVaultRole($conn, $vaultId, 'Owner');

    if ($_POST['action'] === 'addPermission' && isset($_POST['user_id'], $_POST['role_id'])) {
        $userId = (int)$_POST['user_id'];
        $roleId = (int)$_POST['role_id'];
        if (!in_array($roleId, [1, 2, 3], true)) {
            http_response_code(400);
            die('Invalid role.');
        }
        $stmt = $conn->prepare("INSERT INTO vault_permissions (user_id, role_id, vault_id) VALUES (?, ?, ?)");
        $stmt->bind_param("iii", $userId, $roleId, $vaultId);
        $stmt->execute();
    }

    if ($_POST['action'] === 'deletePermission' && isset($_POST['deletePermissionId'])) {
        $permissionId = (int)$_POST['deletePermissionId'];
        // Scoped to this vault, so a permission ID from another vault can't be deleted from here.
        $stmt = $conn->prepare("DELETE FROM vault_permissions WHERE permission_id = ? AND vault_id = ?");
        $stmt->bind_param("ii", $permissionId, $vaultId);
        $stmt->execute();
    }

    header("Location: vault_permissions.php?vault_id=$vaultId");
    exit();
}

$selectedVaultId = isset($_GET['vault_id']) ? (int)$_GET['vault_id'] : 0;
if ($selectedVaultId <= 0) {
    die('Invalid vault.');
}
requireVaultRole($conn, $selectedVaultId, 'Owner');

$stmt = $conn->prepare("SELECT vault_name FROM vaults WHERE vault_id = ?");
$stmt->bind_param("i", $selectedVaultId);
$stmt->execute();
$selectedVaultName = $stmt->get_result()->fetch_assoc()['vault_name'] ?? '';

$stmt = $conn->prepare(
    "SELECT u.username, r.role, p.permission_id, u.user_id, r.role_id
       FROM vault_permissions p
       JOIN users u ON p.user_id = u.user_id
       JOIN roles r ON p.role_id = r.role_id
      WHERE p.vault_id = ?"
);
$stmt->bind_param("i", $selectedVaultId);
$stmt->execute();
$permissions = $stmt->get_result()->fetch_all(MYSQLI_ASSOC);

// Dropdown data. Only usernames and IDs are needed, never password hashes.
$resultUsers = $conn->query("SELECT user_id, username FROM users WHERE approved = 1 ORDER BY username");
$resultRoles = $conn->query("SELECT role_id, role FROM roles");
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>User-Role-Vault Relationship Management</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <!-- Bootstrap JS and other scripts -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
</head>
</head>

<?php include '../components/nav-bar.php' ?>

<body>

    <div class="container mt-4">
        <?php if ($selectedVaultId): ?>
            <h3>Permissions for Vault:
                <?php echo htmlspecialchars((string)$selectedVaultName, ENT_QUOTES); ?>
            </h3>

            <table class="table">
                <thead>
                    <tr>
                        <th>User</th>
                        <th>Role</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($permissions as $permission): ?>
                        <tr>
                            <td>
                                <?php echo htmlspecialchars((string)$permission['username'], ENT_QUOTES); ?>
                            </td>
                            <td>
                                <?php echo htmlspecialchars((string)$permission['role'], ENT_QUOTES); ?>
                            </td>
                            <td>


                                <!-- Delete button to open the delete modal -->
                                <button class="btn btn-danger btn-sm delete-btn" data-toggle="modal"
                                    data-target="#deletePermissionModal"
                                    data-delete-permission-id="<?php echo htmlspecialchars((string)$permission['permission_id'], ENT_QUOTES); ?>"
                                    data-vault_id="<?php echo htmlspecialchars((string)$selectedVaultId, ENT_QUOTES); ?>">
                                    Delete
                                </button>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>

            <!-- Add Permission button to open a modal for adding a new permission -->
            <button class="btn btn-success" data-toggle="modal" data-target="#addModal">Add Permission</button>


            <!-- Delete Permission Modal -->
            <div class="modal fade" id="deletePermissionModal" tabindex="-1" role="dialog"
                aria-labelledby="deletePermissionModalLabel" aria-hidden="true">
                <div class="modal-dialog" role="document">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="deleteUserModalLabel">Delete User</h5>
                            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                <span aria-hidden="true">&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <p>Are you sure you want to delete this user's permission?</p>
                            <!-- Add your form for deleting a user here -->
                            <form id="deleteForm" method="post" action="<?php echo htmlspecialchars((string)$_SERVER['PHP_SELF'], ENT_QUOTES); ?>">
                                <input type="hidden" name="vault_id" value="<?php echo htmlspecialchars((string)$selectedVaultId, ENT_QUOTES); ?>">
                                <input type="hidden" name="deletePermissionId" id="deletePermissionId" value="">
                                <input type="hidden" name="action" value="deletePermission">
                                <button type="submit" class="btn btn-danger">Delete Permission</button>
                            </form>

                            <script>
                                $(document).ready(function () {
                                    // Attach click event to delete buttons
                                    $('.delete-btn').click(function () {
                                        var deletePermissionId = $(this).data('delete-permission-id');                                        
                                        // Set values in the delete form
                                        $('#deletePermissionId').val(deletePermissionId);                                        
                                        // // Submit the delete form
                                        // $('#deleteForm').submit();
                                    });
                                });
                            </script>
                        </div>
                    </div>
                </div>
            </div>




            <!-- Add Permission Modal -->
            <div class="modal fade" id="addModal" tabindex="-1" role="dialog" aria-labelledby="addModalLabel"
                aria-hidden="true">
                <div class="modal-dialog" role="document">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="addModalLabel">Add Permission</h5>
                            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                <span aria-hidden="true">&times;</span>
                            </button>
                        </div>
                        <div class="modal-body">
                            <!-- Add your form for adding permission here -->
                            <form method="post" action="<?php echo htmlspecialchars((string)$_SERVER['PHP_SELF'], ENT_QUOTES); ?>">
                                <input type="hidden" name="vault_id" value="<?php echo htmlspecialchars((string)$selectedVaultId, ENT_QUOTES); ?>">
                                <input type="hidden" name="action" value="addPermission">
                                <div class="form-group">
                                    <label for="user">Select User:</label>
                                    <select class="form-control" id="user" name="user_id" required>
                                        <?php mysqli_data_seek($resultUsers, 0); ?>
                                        <?php while ($user = $resultUsers->fetch_assoc()): ?>
                                            <option value="<?php echo htmlspecialchars((string)$user['user_id'], ENT_QUOTES); ?>">
                                                <?php echo htmlspecialchars((string)$user['username'], ENT_QUOTES); ?>
                                            </option>
                                        <?php endwhile; ?>
                                    </select>
                                </div>

                                <div class="form-group">
                                    <label for="role">Select Role:</label>
                                    <select class="form-control" id="role" name="role_id" required>
                                        <?php mysqli_data_seek($resultRoles, 0); ?>
                                        <?php while ($role = $resultRoles->fetch_assoc()): ?>
                                            <option value="<?php echo htmlspecialchars((string)$role['role_id'], ENT_QUOTES); ?>">
                                                <?php echo htmlspecialchars((string)$role['role'], ENT_QUOTES); ?>
                                            </option>
                                        <?php endwhile; ?>
                                    </select>
                                </div>

                                <button type="submit" class="btn btn-primary">Add Permission</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>

        <?php endif; ?>
    </div>

</body>


<?php
$conn->close();
?>