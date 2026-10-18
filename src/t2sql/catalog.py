"""Hand-written Chinook knowledge, shared by the scope classifier and the schema index (RAG)."""

# One line per table. Unknown tables show their name only.
TABLE_DESCRIPTIONS = {
    "album": "music albums; each album belongs to one artist",
    "artist": "artists / bands (artistas, musicians); an artist has many albums",
    "customer": "customers / clients (clientes) of the store (name, company, city, country, email, assigned support rep)",
    "employee": "store employees (empleados; sales support agents, managers) with title, hire date and reporting hierarchy",
    "genre": "music genres (géneros) such as Rock, Jazz, Metal",
    "invoice": "sales invoices / purchases: customer, invoice date, billing address and country, total amount; revenue and sales per country or period (purchases, compras, ventas, facturas, facturación)",
    "invoice_line": "line items of each invoice (purchased items, tracks bought / canciones compradas): unit price and quantity",
    "media_type": "file formats of tracks, e.g. MPEG audio file, AAC",
    "playlist": "named playlists, e.g. Music, Classical, 90's Music",
    "playlist_track": "link table: which tracks belong to which playlist (many-to-many)",
    "track": "individual songs / tracks (canciones): name, album, genre, media type, composer, duration, size, unit price",
}

# Everyday words (English / Spanish) that users use for each table's concepts. Appended to the
# table's document in the schema index only (RAG vocabulary), not shown to the SQL generator.
# Generic domain vocabulary on purpose: no evaluation question is copied here (see docs/eval-findings.md).
TABLE_KEYWORDS = {
    "album": "album, record, disc, release; álbum, álbumes, disco, discos",
    "artist": "artist, band, singer, musician, performer, group; artista, banda, cantante, músico, grupo",
    "customer": "customer, client, buyer, shopper; who bought or purchased something; cliente, comprador, quién compró, quiénes compraron, país del cliente",
    "employee": "employee, staff, worker, agent, manager, supervisor, salesperson, hired; empleado, personal, agente, gerente, jefe, vendedor, contratado",
    "genre": "genre, style, kind of music, category; género, género musical, estilo",
    "invoice": "purchase, order, sale, bill, receipt, payment, revenue, income, sales, spending, spent, amount billed; compra, venta, factura, facturación, ingresos, ventas, gasto, gastó, importe, pago, por año, por mes",
    "invoice_line": "items sold, units sold, quantity sold, best-selling, most sold, what was bought or purchased, revenue per track, genre, artist or album; ítems vendidos, unidades vendidas, más vendido, vendidas, lo que se compró, ingresos por canción, género, artista o álbum",
    "media_type": "format, file type, codec, audio or video file; formato, tipo de archivo, tipo de medio",
    "playlist": "playlist, list of songs, collection; lista de reproducción",
    "playlist_track": "songs in a playlist, playlist contents; canciones de una playlist",
    "track": "song, track, music, title, length, duration, price; canción, canciones, tema, temas, duración, precio",
}

# "table.column" -> hint. Only where the name alone is ambiguous or the meaning is not obvious.
COLUMN_HINTS = {
    "invoice.total": "money amount of the whole invoice (revenue, sales, billed)",
    "invoice.invoice_date": "date and time of the purchase (year, month)",
    "invoice.billing_country": "country where the invoice was billed (sales by country)",
    "invoice.customer_id": "the customer who made the purchase",
    "invoice_line.unit_price": "price of one unit of the track; line revenue is unit_price * quantity",
    "invoice_line.quantity": "number of units bought in the line; line revenue is unit_price * quantity",
    "invoice_line.track_id": "the track that was sold; links invoices to tracks, albums, artists and genres",
    "track.milliseconds": "duration (length) of the track in milliseconds",
    "track.bytes": "file size of the track in bytes",
    "track.unit_price": "list price of the track",
    "track.composer": "composer / songwriter of the track (free text, often null)",
    "track.name": "title of the song",
    "track.genre_id": "genre of the track",
    "track.album_id": "album the track belongs to; reaches the artist through album",
    "album.title": "title of the album",
    "album.artist_id": "the artist who made the album",
    "artist.name": "name of the artist or band",
    "genre.name": "genre name, e.g. Rock, Jazz",
    "playlist.name": "playlist name, e.g. Music, Movies, Classical",
    "playlist_track.playlist_id": "playlist that contains the track",
    "playlist_track.track_id": "track contained in the playlist",
    "media_type.name": "file format name",
    "customer.support_rep_id": "employee (sales support agent) assigned to the customer",
    "customer.country": "country of residence of the customer",
    "customer.company": "company the customer works for (often null)",
    "employee.reports_to": "employee_id of the manager (self-reference to employee)",
    "employee.title": "job title, e.g. General Manager, Sales Support Agent",
    "employee.hire_date": "date the employee was hired",
    "employee.birth_date": "date of birth of the employee",
}
