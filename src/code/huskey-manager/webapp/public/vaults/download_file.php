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

if (!isset($_GET['file'], $_GET['vault_id'])) {
    die('Invalid file request.');
}

$vaultId = (int)$_GET['vault_id'];
$requestedFile = basename($_GET['file']); // never trust a path from the client

// 1. The user must hold a role on this vault.
requireVaultRole($conn, $vaultId, 'Viewer');

// 2. The file must be one recorded against a password in this vault. Only the stored
//    path is ever opened, so ../ or absolute paths in ?file= can't reach anything else.
$stmt = $conn->prepare("SELECT file_path FROM vault_passwords WHERE vault_id = ? AND file_path IS NOT NULL");
$stmt->bind_param("i", $vaultId);
$stmt->execute();
$filePath = null;
foreach ($stmt->get_result()->fetch_all(MYSQLI_ASSOC) as $row) {
    if (basename($row['file_path']) === $requestedFile) {
        $filePath = $row['file_path'];
        break;
    }
}

if ($filePath === null) {
    $logger->warning("Blocked file download", ['username' => $_SESSION['authenticated'], 'vault_id' => $vaultId, 'file' => $_GET['file']]);
    http_response_code(403);
    die('You do not have permission to access this file.');
}
if (!is_file($filePath)) {
    http_response_code(404);
    die('File not found.');
}

header('Content-Description: File Transfer');
header('Content-Type: application/octet-stream');
header('Content-Disposition: attachment; filename="' . basename($filePath) . '"');
header('X-Content-Type-Options: nosniff');
header('Cache-Control: no-store');
header('Content-Length: ' . filesize($filePath));
readfile($filePath);

$logger->info("File downloaded", ['username' => $_SESSION['authenticated'], 'vault_id' => $vaultId, 'file' => basename($filePath)]);
exit();
