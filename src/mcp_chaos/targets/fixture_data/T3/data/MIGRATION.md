# Migrating config.json to config.v2.json

Apply exactly these three rules. Every other key keeps its name, value and type.

1. Rename the key `app_name` to `service_name` (same value).
2. Nest `db_host`, `db_port` and `db_name` under a new object `database`,
   renaming them to `host`, `port` and `name` respectively (values unchanged).
3. Change `timeout` from a string to an integer number of seconds
   (for example `"30"` becomes `30`).

Write the result to `data/config.v2.json`. Leave `data/config.json` as it is.
