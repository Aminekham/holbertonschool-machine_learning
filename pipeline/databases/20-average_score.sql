-- Averaging the student scores
DELIMITER //
CREATE PROCEDURE ComputeAverageScoreForUser(IN user_id INT)
BEGIN
    DECLARE avg_score FLOAT;
    SELECT IFNULL(AVG(score), 0) INTO avg_score
    FROM corrections
    WHERE user_id = user_id;
    UPDATE users
    SET average_score = avg_score
    WHERE id = user_id;
END//
DELIMITER ;
