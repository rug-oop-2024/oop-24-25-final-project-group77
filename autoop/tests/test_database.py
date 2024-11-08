import unittest

from autoop.core.database import Database
from autoop.core.storage import LocalStorage
import random
import tempfile


class TestDatabase(unittest.TestCase):
    """ Test Database class """
    def setUp(self) -> None:
        """
        Set up a test database instance with a temporary storage location.
        The database is initialized before each test method.
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self) -> None:
        """
        Tests that the database is correctly initialized
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self) -> None:
        """
        Tests that the set method saves the entry to the database
        and that we can retrieve it from the database.
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self) -> None:
        """
        Tests that the delete method removes the entry from the database
        and from the persisted file.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self) -> None:
        """
        Test the persistence of data in the Database class.

        This test ensures that data persisted in the database remains
        consistent across different Database instances.
        It sets a value in the database, creates a new instance of the
        Database with the same storage, and verifies
        that the value can be retrieved from the new instance.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self) -> None:
        """
        Test the refresh method of the Database class.

        This test verifies that the refresh method correctly reloads
        the data from storage into a new Database instance. It sets
        a key-value pair in the original database and refreshes the
        other_db instance to ensure the data is synced and accessible
        after the refresh.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self) -> None:
        """ Test if list returns the correct entries. """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))
