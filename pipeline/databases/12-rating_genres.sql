-- genres by ratings
SELECT tg.name AS name, SUM(tsr.rate) AS rating FROM tv_genres tg
JOIN tv_show_genres tsg ON tg.id = tsg.genre_id
JOIN tv_show_ratings tsr ON tsg.show_id = tsr.show_id
GROUP BY tg.name ORDER BY rating DESC;
