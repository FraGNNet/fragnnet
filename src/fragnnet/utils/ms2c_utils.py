import sqlite3

from tqdm import tqdm


class MolCandidateDB:
    def __init__(self, db_config: str):
        self.conn = None
        if type(db_config) == str:
            self.db_file = db_config
        else:
            self.db_file = db_config.db_file

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # save cache on exit
        self.disconnect()

    def connect(self):
        self.disconnect()
        # Try to connect
        # conn = None
        try:
            self.conn = sqlite3.connect(
                self.db_file, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES
            )
        except:
            print(f"Unable to connect to the database: {self.db_file} ")
        else:
            # print("Database Connection Established")
            # use wal model
            self.conn.execute("pragma journal_mode=wal2")

    def disconnect(self):
        if self.conn is not None:
            self.conn.cursor().close()
            self.conn.close()
            self.conn = None

    def check_connection(self):
        if self.conn is None:
            raise ValueError("No Connection")

    def execute_query(self, query: str, commit: bool = False, fetch: bool = True):
        self.check_connection()
        fetched = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
        except Exception as err:
            # pass exception to function
            print(f"{err}")
            self.conn.rollback()
            return None
        else:
            if fetch:
                fetched = cursor.fetchall()
            if commit:
                self.conn.commit()
        # print("size of fetched", sys.getsizeof(fetched) >> 20, "MB")
        return fetched

    def execute_many_query(self, query: str, data):
        self.check_connection()
        try:
            cursor = self.conn.cursor()
            cursor.executemany(query, data)
        except Exception as err:
            # pass exception to function
            # DatabaseUtilities.print_psycopg2_exception(err)
            print(f"{err}")
            print(query)
            print(data)
            self.conn.rollback()
            return None
        else:
            self.conn.commit()

    def _go_fast_at_all_cost(self):
        """_summary_"""
        # Turning off journal_mode will result in no rollback journal
        self.conn.execute("PRAGMA journal_mode = OFF;")
        # SQLite will not care about writing to disk reliably and hands off that responsibility to the OS
        self.conn.execute("PRAGMA synchronous = 0;")
        # The cache_size specifies how many memory pages SQLite is allowed to hold in the memory
        self.conn.execute("PRAGMA cache_size = 4000000;")  # give it  4GB
        self.conn.execute("PRAGMA locking_mode = EXCLUSIVE;")
        self.conn.execute("PRAGMA temp_store = MEMORY;")

    def _create_compound_table(self):
        query = """
            CREATE TABLE IF NOT EXISTS compound (
                id INTEGER primary key,
                inchikey TEXT,
                smiles TEXT,
                formula TEXT,
                exact_mass FLOAT);
            """
        self.execute_query(query)

        for index_col in ["exact_mass"]:
            self.execute_query(
                f"CREATE INDEX IF NOT EXISTS {index_col}_idx on compound ({index_col});"
            )

    def create_tables(self):
        self._create_compound_table()

    def get_compounds_by_exact_mass_range(
        self, mw_min: float, mw_max: float, verbose: bool = False
    ):
        """_summary_

        Args:
            smiles_list (List[str]): _description_

        Returns:
            _type_: return -1 for not found
        """

        selection_query = f"SELECT COALESCE(id,-1),inchikey,smiles,formula,exact_mass FROM compound WHERE exact_mass >= {mw_min} AND exact_mass <= {mw_max}"
        if verbose:
            print(f">get_compounds_by_colname {selection_query}")
        return self.execute_query(selection_query, fetch=True)

    def get_compounds_by_colname(
        self, col_name: str, values_list: list[str | int | float], verbose: bool = False
    ):
        """_summary_

        Args:
            smiles_list (List[str]): _description_

        Returns:
            _type_: return -1 for not found
        """
        # TODO use fetch and yelid if there is too many returns
        if col_name in ["id", "exact_mass"]:
            values_query = ",".join([f"{i}" for i in values_list])
        elif col_name in ["inchikey", "smiles", "formula"]:
            values_query = ",".join([f"'{i}'" for i in values_list])
        else:
            raise AttributeError(
                f"col_name need to be one of 'id','exact_mass','inchikey','smiles','formula' not {col_name} "
            )

        selection_query = f"SELECT COALESCE(id,-1),inchikey,smiles,formula,exact_mass FROM compound WHERE {col_name} in ({values_query})"
        if verbose:
            print(f">get_compounds_by_colname {selection_query}")
        return self.execute_query(selection_query, fetch=True)

    @classmethod
    def _list_to_sqlite(cls, data: list, quote=True):
        if type(data) is list:
            if quote:
                return "('" + "','".join([str(d) for d in data]) + "')"
            else:
                return f"({','.join([str(d) for d in data])})"
        else:
            if quote:
                return f"('{data}')"
            else:
                return f"({data})"

    def add_compounds_from_df(self, df, chunk_size=10000):
        compound_query = """INSERT OR IGNORE INTO compound(id, inchikey, exact_mass, formula, smiles) VALUES (?,?,?,?,?); """
        for i in tqdm(range(0, df.shape[0], chunk_size), leave=False):
            compound_insert_data = df[i : i + chunk_size].values.tolist()
            self.execute_many_query(compound_query, compound_insert_data)
