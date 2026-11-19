<?php
// Server-side authorization for vault actions.
// Every read or write on a vault goes through these checks, so changing an ID in the URL
// or in a hidden form field can't reach a vault (or a password row) the user doesn't hold a role on.

const VAULT_ROLE_RANK = ['Viewer' => 1, 'Editor' => 2, 'Owner' => 3];

/** The signed-in user's role on a vault ('Owner' | 'Editor' | 'Viewer'), or null if none. */
function vaultRole(mysqli $conn, int $vaultId): ?string
{
    if (!empty($_SESSION['isSiteAdministrator'])) {
        return 'Owner';
    }
    $stmt = $conn->prepare(
        "SELECT r.role
           FROM vault_permissions vp
           JOIN users u ON u.user_id = vp.user_id
           JOIN roles r ON r.role_id = vp.role_id
          WHERE vp.vault_id = ? AND u.username = ?
          ORDER BY vp.role_id ASC
          LIMIT 1"
    );
    $stmt->bind_param("is", $vaultId, $_SESSION['authenticated']);
    $stmt->execute();
    $row = $stmt->get_result()->fetch_assoc();
    $stmt->close();
    return $row['role'] ?? null;
}

/** Stop with 403 unless the user holds at least $minimumRole on the vault. Returns the role. */
function requireVaultRole(mysqli $conn, int $vaultId, string $minimumRole): string
{
    $role = vaultRole($conn, $vaultId);
    if ($role === null || VAULT_ROLE_RANK[$role] < VAULT_ROLE_RANK[$minimumRole]) {
        if (isset($GLOBALS['logger'])) {
            $GLOBALS['logger']->warning("Unauthorized vault action", [
                'username' => $_SESSION['authenticated'] ?? null,
                'vault_id' => $vaultId,
                'required' => $minimumRole,
            ]);
        }
        http_response_code(403);
        die($role === null
            ? 'You do not have permission to access this vault.'
            : 'You do not have permission to perform this action.');
    }
    return $role;
}

/** Stop with 403 unless the password row actually belongs to the vault being acted on. */
function requirePasswordInVault(mysqli $conn, int $passwordId, int $vaultId): array
{
    $stmt = $conn->prepare("SELECT * FROM vault_passwords WHERE password_id = ? AND vault_id = ?");
    $stmt->bind_param("ii", $passwordId, $vaultId);
    $stmt->execute();
    $row = $stmt->get_result()->fetch_assoc();
    $stmt->close();
    if (!$row) {
        http_response_code(403);
        die('You do not have permission to perform this action.');
    }
    return $row;
}

/**
 * Save an uploaded attachment safely, or return null if none was sent.
 * Allow-listed extensions only, size-capped, stored under a random name so it can't
 * overwrite another file or be requested as a script (nginx also denies /vaults/uploads/).
 */
function storeUpload(array $file, string $uploadDir): ?string
{
    if (empty($file['name'])) {
        return null;
    }
    $allowed = ['pdf', 'txt', 'csv', 'png', 'jpg', 'jpeg', 'docx', 'xlsx'];
    $ext = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    if ($file['error'] !== UPLOAD_ERR_OK || !in_array($ext, $allowed, true) || $file['size'] > 5 * 1024 * 1024) {
        http_response_code(400);
        die('Upload rejected: allowed types are ' . implode(', ', $allowed) . ' (max 5 MB).');
    }
    $stem = preg_replace('/[^A-Za-z0-9_-]/', '_', pathinfo($file['name'], PATHINFO_FILENAME));
    $dest = $uploadDir . bin2hex(random_bytes(8)) . '_' . substr($stem, 0, 40) . '.' . $ext;
    if (!move_uploaded_file($file['tmp_name'], $dest)) {
        die('Error uploading file.');
    }
    return $dest;
}
