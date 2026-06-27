try:
    from sqlalchemy import Column, DateTime, Integer, String, Text
    from sqlalchemy.orm import declarative_base

    Base = declarative_base()

    class Run(Base):
        __tablename__ = "runs"

        id = Column(Integer, primary_key=True)
        run_id = Column(String, unique=True, nullable=False)
        task = Column(String, nullable=False)
        model_type = Column(String, nullable=False)
        status = Column(String, nullable=False)
        config_path = Column(String, nullable=False)
        output_dir = Column(String, nullable=False)
        metrics_json = Column(Text)
        report_path = Column(String)
        created_at = Column(DateTime, nullable=False)
        updated_at = Column(DateTime, nullable=False)

    class Artifact(Base):
        __tablename__ = "artifacts"

        id = Column(Integer, primary_key=True)
        run_id = Column(String, nullable=False)
        artifact_type = Column(String, nullable=False)
        file_path = Column(String, nullable=False)
        created_at = Column(DateTime, nullable=False)

except Exception:
    Base = None
    Run = None
    Artifact = None
