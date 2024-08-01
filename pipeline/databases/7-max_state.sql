-- The max temperature of each state
SELECT state, MAX(value) FROM temperatures
GROUP BY state