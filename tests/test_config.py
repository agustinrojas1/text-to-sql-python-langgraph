from t2sql.config import Settings


def test_settings_load_from_dotenv_file(tmp_path, monkeypatch):
    # pydantic-settings reads the file itself; python-dotenv is not needed.
    for var in ("T2SQL_MAX_ROWS", "GOOGLE_API_KEY", "T2SQL_GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    env = tmp_path / ".env"
    env.write_text("T2SQL_MAX_ROWS=7\nGOOGLE_API_KEY=fake-test-key\nUNRELATED=1\n")
    s = Settings(_env_file=env)
    assert s.max_rows == 7 and s.google_api_key == "fake-test-key"


def test_default_env_file_is_dotenv_in_cwd(tmp_path, monkeypatch):
    monkeypatch.delenv("T2SQL_MAX_ROWS", raising=False)
    (tmp_path / ".env").write_text("T2SQL_MAX_ROWS=9\n")
    monkeypatch.chdir(tmp_path)
    assert Settings().max_rows == 9


def test_embedding_cache_dir_precedence(tmp_path, monkeypatch):
    from t2sql.config import default_embedding_cache_dir

    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_embedding_cache_dir() == str(tmp_path / "xdg" / "t2sql" / "fastembed")

    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert default_embedding_cache_dir() == str(tmp_path / "home" / ".cache" / "t2sql" / "fastembed")

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path / "fe"))
    assert default_embedding_cache_dir() == str(tmp_path / "fe")


def test_embedding_cache_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("T2SQL_EMBEDDING_CACHE_DIR", str(tmp_path / "c"))
    assert Settings(_env_file=None).embedding_cache_dir == str(tmp_path / "c")
