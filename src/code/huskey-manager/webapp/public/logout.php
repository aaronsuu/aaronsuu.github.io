<?php
session_start();
include './components/loggly-logger.php';

$username = $_SESSION['authenticated'] ?? 'unknown';
$time = date("Y-m-d H:i:s");
$logger->info("User $username logged out at $time");

// Destroy the server-side session and expire the session cookie,
// so a copied PHPSESSID can't be replayed after logout.
$_SESSION = [];
$params = session_get_cookie_params();
setcookie(session_name(), '', time() - 3600, $params['path'], $params['domain'], $params['secure'], $params['httponly']);
session_destroy();
// Redirect to the login page
header('Location: login.php');
exit();
?>