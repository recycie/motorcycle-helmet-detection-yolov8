-- CREATE DATABASE helmet;
CREATE TABLE videos (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(32) NOT NULL,
    description VARCHAR(100),
    source TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE detection (
    id VARCHAR(128) PRIMARY KEY,
    motorcycle_id VARCHAR(32) NOT NULL,
    source_id INT NOT NULL,
    driver BOOLEAN DEFAULT 0,
    helmet BOOLEAN DEFAULT 0,
    helmet_score DECIMAL(4, 3),
    person_score DECIMAL(4, 3),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES videos(id)
);

CREATE TABLE bbox (
    detection_id VARCHAR(128) PRIMARY KEY,
    helmet_x1 VARCHAR(8),
    helmet_y1 VARCHAR(8),
    helmet_x2 VARCHAR(8),
    helmet_y2 VARCHAR(8),
    person_x1 VARCHAR(8) NOT NULL,
    person_y1 VARCHAR(8) NOT NULL,
    person_x2 VARCHAR(8) NOT NULL,
    person_y2 VARCHAR(8) NOT NULL,
    FOREIGN KEY (detection_id) REFERENCES detection(id) ON DELETE CASCADE
);
