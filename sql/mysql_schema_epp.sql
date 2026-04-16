USE deteccion_epp;

CREATE TABLE IF NOT EXISTS epp_events (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_id VARCHAR(150) NOT NULL UNIQUE,
    camera_id VARCHAR(100) NOT NULL,
    scenario_id VARCHAR(100) NOT NULL,
    zone_name VARCHAR(100) NULL,
    person_track_id VARCHAR(100) NOT NULL,

    event_type ENUM('violation_started', 'violation_resolved') NOT NULL,
    status ENUM('compliant', 'non_compliant') NOT NULL,
    observed_status ENUM('compliant', 'non_compliant') NOT NULL,

    frame_number INT NOT NULL,
    event_observed_at DATETIME(6) NOT NULL,
    event_confirmed_at DATETIME(6) NOT NULL,

    helmet_ok TINYINT(1) NOT NULL DEFAULT 0,
    vest_ok TINYINT(1) NOT NULL DEFAULT 0,
    helmet_score DECIMAL(6,4) NULL,
    vest_score DECIMAL(6,4) NULL,

    person_box_json LONGTEXT NULL,
    head_box_json LONGTEXT NULL,
    torso_box_json LONGTEXT NULL,
    helmet_box_json LONGTEXT NULL,
    vest_box_json LONGTEXT NULL,

    violation_codes_json LONGTEXT NOT NULL,
    confirmed_status_snapshot_json LONGTEXT NULL,

    model_version VARCHAR(255) NULL,

    resolved_by_event_id VARCHAR(150) NULL,
    resolved_at DATETIME(6) NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_camera_id (camera_id),
    INDEX idx_person_track_id (person_track_id),
    INDEX idx_event_confirmed_at (event_confirmed_at),
    INDEX idx_event_type (event_type),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS epp_event_evidence (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    event_id VARCHAR(150) NOT NULL,
    image_full_path TEXT NULL,
    image_annotated_path TEXT NULL,
    image_crop_path TEXT NULL,
    video_path TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_epp_event_evidence_event
        FOREIGN KEY (event_id) REFERENCES epp_events(event_id)
        ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;