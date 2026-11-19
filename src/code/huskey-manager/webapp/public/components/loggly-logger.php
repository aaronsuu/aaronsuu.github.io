<?php

require_once __DIR__ . '/../../vendor/autoload.php';

use Monolog\Logger;
use Monolog\Handler\LogglyHandler;
use Monolog\Handler\StreamHandler;

$logger = new Logger('HusKey Manager');

// Ship logs to Loggly when a token is configured; otherwise write to stderr,
// which shows up in `docker compose logs backend-php-server`.
$logglyToken = getenv('LOGGLY_TOKEN') ?: ($_ENV['LOGGLY_TOKEN'] ?? '');

if ($logglyToken !== '') {
    $logger->pushHandler(new LogglyHandler($logglyToken . '/tag/monolog', Logger::INFO));
} else {
    $logger->pushHandler(new StreamHandler('php://stderr', Logger::INFO));
}
