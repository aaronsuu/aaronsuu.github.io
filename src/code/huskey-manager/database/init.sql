-- Drop tables if they exist
-- DROP TABLE IF EXISTS vault_permissions;
-- DROP TABLE IF EXISTS vault_passwords;
-- DROP TABLE IF EXISTS vaults;
-- DROP TABLE IF EXISTS users;
-- DROP TABLE IF EXISTS roles;

-- Create the database
CREATE DATABASE IF NOT EXISTS password_manager;

-- Use the database
USE password_manager;

-- Create the roles table
CREATE TABLE IF NOT EXISTS roles (
    role_id INT AUTO_INCREMENT PRIMARY KEY,
    role ENUM('Owner', 'Editor', 'Viewer') NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    first_name VARCHAR(255) NOT NULL,
    last_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    default_role_id INT,    
    approved INT,
    FOREIGN KEY (default_role_id) REFERENCES roles(role_id) ON DELETE CASCADE 
);

-- Failed login attempts, tracked server-side so clearing cookies cannot reset the count
CREATE TABLE IF NOT EXISTS login_attempts (
    attempt_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL,
    ip_address VARCHAR(45) NOT NULL,
    attempted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_attempt_user (username, attempted_at),
    INDEX idx_attempt_ip (ip_address, attempted_at)
);

-- Create the vaults table
CREATE TABLE IF NOT EXISTS vaults (
    vault_id INT AUTO_INCREMENT PRIMARY KEY,    
    vault_name VARCHAR(255) NOT NULL    
);

-- Create the passwords table
CREATE TABLE IF NOT EXISTS vault_passwords (
    password_id INT AUTO_INCREMENT PRIMARY KEY,
    vault_id INT,
    username VARCHAR(255) NOT NULL,
    website VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    notes TEXT,
    file_path VARCHAR(512), -- New column for file path
    FOREIGN KEY (vault_id) REFERENCES vaults(vault_id) ON DELETE CASCADE
);

-- Create the permissions table
CREATE TABLE IF NOT EXISTS vault_permissions (
    permission_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    vault_id INT,
    role_id INT,    
    FOREIGN KEY (vault_id) REFERENCES vaults(vault_id) ON DELETE CASCADE,   
    FOREIGN KEY (role_id) REFERENCES roles(role_id) ON DELETE CASCADE        
);

-- Create an index on username for faster retrieval
CREATE UNIQUE INDEX idx_username ON users(username);

-- Insert sample roles
INSERT INTO roles (role_id, role)
VALUES
    (1, 'Owner'),
    (2, 'Editor'),
    (3, 'Viewer');

-- Insert sample users
-- Passwords are stored as bcrypt hashes (password_hash / PASSWORD_BCRYPT).
INSERT INTO users (user_id, username, first_name, last_name, email, password, default_role_id, approved)
VALUES
    (1, 'username', 'User', 'Name', 'user@info310.net', '$2y$12$dIwcO9.zxAOWp7fn/rn9u.qoXPOw4HTYAtL.rzK6zzdbSLZODaEuC', 3, 1),
    (2, 'admin', 'Super', 'Admin', 'admin@info310.com', '$2y$12$KQKQqcPofhHU90HWeVXRgO9qvZ7/oZlN.NHSa6LHRuINnkKHPm2Iq', 1, 1),
    (3, 'johndoe', 'John', 'Doe', 'john.doe@info310.com', '$2y$12$.6gSklVegXsJRfuGvGIki.WjWMvA5hI.I1wXpyKvU88tv3KdDtbLe', 3, 1),
    (4, 'janedoe', 'Jane', 'Doe', 'jane.doe@info310.com', '$2y$12$p/jgUp5/9VrSElitoLllIOxKdE9.BEbBVmkj.qDtN0lfLg6kBJ./y', 3, 1),
    (5, 'alice_smith', 'Alice', 'Smith', 'alice.smith@info310.com', '$2y$12$ziL3LX8S3.mQDFsPZFTQP.krl0mntvYVhNLg3hb/UmniCbA0GY/92', 3, 1),
    (6, 'bob_johnson', 'Bob', 'Johnson', 'bob.johnson@info310.com', '$2y$12$2SMoa5NmzO7VXYT830eoqeooRRR3.sqZf79JcLjQp9/SqV5b4OI1W', 3, 1),
    (7, 'sarah_wilson', 'Sarah', 'Wilson', 'sarah.wilson@info310.com', '$2y$12$.C1qYOKoT4MGG7HKrdvwDu/4vBS39Ub6PK/pMpO87YCTjsYTpWTly', 3, 1),
    (8, 'mike_brown', 'Mike', 'Brown', 'mike.brown@info310.com', '$2y$12$gn/O63UPR4fq8qWIabVa.OY95uVShoos5z0i49JkskO/Trve7NhY2', 3, 1),
    (9, 'emily_davis', 'Emily', 'Davis', 'emily.davis@info310.com', '$2y$12$ZjxYcXzQcMRLILqnNhu0auDc2NcU7akC0AflGtseaJGLEoHuCCW2i', 3, 1),
    (10, 'kevin_clark', 'Kevin', 'Clark', 'kevin.clark@info310.com', '$2y$12$OxRCMWPSIKbd3KfsG5ztEOpoMF/JUo1MB98e9wEofFkaHiG//.Ihe', 3, 1),
    (11, 'laura_jones', 'Laura', 'Jones', 'laura.jones@info310.com', '$2y$12$fmBKiIjvWI1/e4blGGZ1O.Ai.m8CxBHDpqHnfz9q6AsvZIPCUUdVO', 3, 1),
    (12, 'chris_miller', 'Chris', 'Miller', 'chris.miller@info310.com', '$2y$12$XHKwQmjlCxj8rYM8x1DR9um9US7KezJPQIAipQkdkaoDcBmqWyyei', 3, 1);


-- Insert sample vaults
INSERT INTO vaults (vault_id, vault_name)
VALUES
    (1, 'Developers Vault'),
    (2, 'Executives Vault'),
    (3, 'HR Vault');

-- Insert sample passwords
INSERT INTO vault_passwords (password_id, vault_id, username, website, password, notes, file_path)
VALUES
    -- Fake logins for "Developers Vault"
    (3, 1, 'developer1', 'github.com', 'dev_password1', 'Developer notes for this password', ''),
    (4, 1, 'developer2', 'stackoverflow.com', 'dev_password2', 'Developer notes for this password', ''),
    (5, 1, 'developer3', 'gitlab.com', 'dev_password3', 'Developer notes for this password', './uploads/dev_notes.pdf'),
    -- Fake logins for "Executives Vault"
    (6, 2, 'executive1', 'companyportal.com', 'exec_password1', 'Executive notes for this password', ''),
    (7, 2, 'executive2', 'boardmeeting.com', 'exec_password2', 'Executive notes for this password', './uploads/exec_notes.txt'),
    -- Fake logins for "HR Vault"
    (8, 3, 'hr1', 'hrportal.com', 'hr_password1', 'HR notes for this password', ''),
    (9, 3, 'hr2', 'payroll.com', 'hr_password2', 'HR notes for this password', ''),
    (10, 3, 'hr3', 'benefits.com', 'hr_password3', 'HR notes for this password', './uploads/hr_notes.pdf');


-- Assign roles and permissions
INSERT INTO vault_permissions (permission_id, user_id, vault_id, role_id)
VALUES
    -- Assigning users to the "Developers Vault"
    (2, 1, 1, 3),  -- Username (Viewer)
    (3, 3, 1, 2),  -- Jane Doe (Editor)
    (4, 5, 1, 3),  -- Alice Smith (Viewer)
    (5, 6, 1, 3),  -- Bob Johnson (Viewer)
    (6, 10, 1, 3), -- Kevin Clark (Viewer)
    (7, 11, 1, 3), -- Laura Jones (Viewer)
    -- Assigning users to the "Executives Vault"
    (8, 2, 2, 1),  -- Super Admin (Owner)
    (9, 7, 2, 2),  -- Sarah Wilson (Editor)
    (10, 8, 2, 3), -- Mike Brown (Viewer)
    -- Assigning users to the "HR Vault"
    (11, 4, 3, 1), -- Jane Doe (Owner)
    (12, 9, 3, 2), -- Emily Davis (Editor)
    (13, 12, 3, 3) -- Chris Miller (Viewer)
