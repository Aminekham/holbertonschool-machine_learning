-- Listing shows with genre
SELECT CONCAT(tv_shows.title, ' - ', tv_show_genres.genre_id) AS result
FROM tv_genres
JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC