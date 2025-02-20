-- Function to divide in SQL
DELIMITER //
CREATE FUNCTION SafeDiv(a INT, b INT) 
RETURNS FLOAT
DETERMINISTIC
BEGIN
    RETURN CASE 
        WHEN b = 0 THEN 0
        ELSE a / b
    END;
END//
DELIMITER ;
