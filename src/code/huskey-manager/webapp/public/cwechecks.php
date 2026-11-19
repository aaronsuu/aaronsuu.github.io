<?php
session_start();

$MAX_SESSION_TIME = 90 * 60; 

if (!isset($_SESSION['authenticated']) || !isset($_SESSION['login_time'])) {
    header("Location: /login.php");
    exit();
}

if (time() - $_SESSION['login_time'] > $MAX_SESSION_TIME) {
    session_unset();
    session_destroy();
    header("Location: /login.php?expired=1");
    exit();
}
