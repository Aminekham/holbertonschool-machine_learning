-- Listing shows with genre
SELECT CONCAT(tv_shows.title, ' - ', tv_show_genres.genre_id) AS result
FROM hbtn_0d_tvshows
JOIN tv_show_genres ON tv_shows.id = tv_show_genres.tv_show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC