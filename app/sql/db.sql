CREATE TABLE detection_api (
    id int PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(48) NOT NULL,
    endpoint VARCHAR(128) NOT NULL,
    apikey VARCHAR(128) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE users (
    id int PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(32) NOT NULL,
    password VARCHAR(255) NOT NULL,
    role VARCHAR(16) DEFAULT 'user' NOT NULL,
    status ENUM('active', 'locked', 'closed', 'suspended') DEFAULT 'active' NOT NULL,
    modify_at DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
