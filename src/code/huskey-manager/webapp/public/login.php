<?php

session_start();
include './components/loggly-logger.php';
$hostname = 'backend-mysql-database';
$username = 'user';
$password = getenv('MYSQL_PASSWORD');
$database = 'password_manager';

$conn = new mysqli($hostname, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

unset($error_message);

// Check if the form is submitted
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    
    $username = $_POST['username'] ?? '';
    $password = $_POST['password'] ?? '';
    $ip = $_SERVER['REMOTE_ADDR'];

    // Never log the submitted password, only who tried and from where.
    $logger->info("Login attempt", ['username' => $username, 'ip' => $ip]);

    $forbiddenChars = '-<>:;\'"{}[]()*=';
    if (strpbrk($username, $forbiddenChars) !== false) {
        $logger->warning("Suspicious login attempt: username contains SQL/HTML metacharacters", ['username' => $username, 'ip' => $ip]);
    }

    // Brute-force lockout. Failures are stored in the database, keyed by account and by IP,
    // so dropping the session cookie does not reset the counter.
    $MAX_ATTEMPTS = 5;
    $LOCKOUT_MINUTES = 15;

    $stmt = $conn->prepare(
        "SELECT
            SUM(username = ?)   AS user_failures,
            SUM(ip_address = ?) AS ip_failures
         FROM login_attempts
         WHERE attempted_at > (NOW() - INTERVAL ? MINUTE)
           AND (username = ? OR ip_address = ?)"
    );
    $stmt->bind_param("ssiss", $username, $ip, $LOCKOUT_MINUTES, $username, $ip);
    $stmt->execute();
    $failures = $stmt->get_result()->fetch_assoc();
    $stmt->close();

    if ((int)$failures['user_failures'] >= $MAX_ATTEMPTS || (int)$failures['ip_failures'] >= $MAX_ATTEMPTS) {
        $logger->warning("Brute force lockout", ['username' => $username, 'ip' => $ip]);
        $error_message = "Too many failed attempts. Try again in $LOCKOUT_MINUTES minutes.";
    } else {
        // Look the user up by username only, then check the password against its bcrypt hash.
        $stmt = $conn->prepare("SELECT * FROM users WHERE username = ? AND approved = 1");
        $stmt->bind_param("s", $username);
        $stmt->execute();
        $userFromDB = $stmt->get_result()->fetch_assoc();
        $stmt->close();

        if ($userFromDB && password_verify($password, $userFromDB['password'])) {
            // Prevent session fixation: issue a fresh session ID on login.
            session_regenerate_id(true);

            $_SESSION['authenticated'] = $username;
            $_SESSION['login_time'] = time();

            if ($userFromDB['default_role_id'] == 1) {
                $_SESSION['isSiteAdministrator'] = 1;
            } else {
                unset($_SESSION['isSiteAdministrator']);
            }

            $stmt = $conn->prepare("DELETE FROM login_attempts WHERE username = ?");
            $stmt->bind_param("s", $username);
            $stmt->execute();

            $logger->info("Login success", ['username' => $username, 'ip' => $ip]);
            header("Location: index.php");
            exit();
        }

        $error_message = 'Invalid username or password.';

        $stmt = $conn->prepare("INSERT INTO login_attempts (username, ip_address) VALUES (?, ?)");
        $stmt->bind_param("ss", $username, $ip);
        $stmt->execute();
        $stmt->close();

        $logger->warning("Login failed", ['username' => $username, 'ip' => $ip]);
    }
    $conn->close();
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <title>Login Page</title>
</head>
<body>
    <div class="container mt-5">
        <div class="col-md-6 offset-md-3">
            <h2 class="text-center">Login</h2>
            <?php if (isset($error_message)) : ?>
                <div class="alert alert-danger" role="alert">
                    <?php echo htmlspecialchars($error_message); ?>
                </div>
            <?php endif; ?>
            <form action="login.php" method="post">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" class="form-control" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" class="form-control" id="password" name="password" required>
                </div>
                <button type="submit" class="btn btn-primary btn-block">Login</button>
            </form>
            <div class="mt-3 text-center">
                <a href="./users/request_account.php" class="btn btn-secondary btn-block">Request an Account</a>
            </div>
        </div>
    </div>
</body>
</html>
