import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
import json
import networkx as nx
import matplotlib.pyplot as plt

class DatabaseSchemaAnalyzer:
    def __init__(self, connection_string):
        """
        Initialize the database schema analyzer
        
        :param connection_string: SQLAlchemy database connection string
        """
        self.engine = create_engine(connection_string)
        self.inspector = inspect(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.schema_info = {
            "database_metadata": {},
            "tables": {},
            "relationships": {
                "foreign_keys": [],
                "graph": {}
            },
            "sample_data": {}
        }

    def analyze_database_metadata(self):
        """
        Collect overall database metadata
        """
        with self.engine.connect() as connection:
            # Database size
            db_size_query = text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as database_size;
            """)
            db_size_result = connection.execute(db_size_query).fetchone()
            
            # Database statistics
            stats_query = text("""
                SELECT 
                    (SELECT count(*) FROM pg_tables WHERE schemaname = 'public') as table_count,
                    (SELECT count(*) FROM pg_indexes WHERE schemaname = 'public') as index_count,
                    (SELECT count(*) FROM pg_views WHERE schemaname = 'public') as view_count
            """)
            stats_result = connection.execute(stats_query).fetchone()

            self.schema_info["database_metadata"] = {
                "database_name": self.engine.url.database,
                "database_size": db_size_result[0],
                "table_count": stats_result[0],
                "index_count": stats_result[1],
                "view_count": stats_result[2]
            }

    def analyze_tables(self):
        """
        Comprehensive analysis of database tables
        """
        for table_name in self.inspector.get_table_names():
            # Detailed column information
            columns = self.inspector.get_columns(table_name)
            
            # Primary keys
            primary_keys = self.inspector.get_primary_keys(table_name)
            
            # Indexes
            indexes = self.inspector.get_indexes(table_name)
            
            # Foreign keys
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            
            # Constraints
            with self.engine.connect() as connection:
                constraints_query = text(f"""
                    SELECT 
                        constraint_name, 
                        constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_name = :table_name AND table_schema = 'public'
                """)
                constraints = connection.execute(constraints_query, {"table_name": table_name}).fetchall()

            # Sample data
            try:
                with self.Session() as session:
                    sample_data_query = text(f"SELECT * FROM {table_name} LIMIT 10")
                    sample_data = session.execute(sample_data_query).fetchall()
                    sample_data_dicts = [dict(row) for row in sample_data]
            except Exception as e:
                sample_data_dicts = [f"Error fetching sample data: {str(e)}"]

            # Store table information
            self.schema_info["tables"][table_name] = {
                "columns": [
                    {
                        "name": col["name"],
                        "type": str(col["type"]),
                        "nullable": col["nullable"],
                        "default": str(col["default"]) if col["default"] is not None else None
                    } for col in columns
                ],
                "primary_keys": primary_keys,
                "indexes": [
                    {
                        "name": idx["name"],
                        "columns": idx["column_names"],
                        "unique": idx["unique"]
                    } for idx in indexes
                ],
                "constraints": [
                    {
                        "name": constraint[0],
                        "type": constraint[1]
                    } for constraint in constraints
                ],
                "sample_data": sample_data_dicts
            }

    def analyze_relationships(self):
        """
        Analyze and visualize database relationships
        """
        # Create a graph of relationships
        G = nx.DiGraph()

        # Collect foreign key relationships
        for table_name in self.inspector.get_table_names():
            foreign_keys = self.inspector.get_foreign_keys(table_name)
            
            for fk in foreign_keys:
                # Record foreign key information
                relationship_info = {
                    "source_table": table_name,
                    "source_columns": fk["constrained_columns"],
                    "target_table": fk["referred_table"],
                    "target_columns": fk["referred_columns"]
                }
                
                self.schema_info["relationships"]["foreign_keys"].append(relationship_info)
                
                # Add nodes and edges to the graph
                G.add_node(table_name)
                G.add_node(fk["referred_table"])
                G.add_edge(table_name, fk["referred_table"], 
                           source_columns=fk["constrained_columns"], 
                           target_columns=fk["referred_columns"])

        # Visualize the relationship graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.5, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=8, font_weight='bold')
        plt.title("Database Table Relationships")
        plt.tight_layout()
        plt.savefig('database_relationships.png')
        plt.close()

        # Store graph structure
        self.schema_info["relationships"]["graph"] = {
            "nodes": list(G.nodes()),
            "edges": [
                {
                    "source": edge[0],
                    "target": edge[1],
                    "source_columns": G.edges[edge].get('source_columns', []),
                    "target_columns": G.edges[edge].get('target_columns', [])
                } for edge in G.edges()
            ]
        }

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive database schema report
        """
        # Analyze different aspects of the database
        self.analyze_database_metadata()
        self.analyze_tables()
        self.analyze_relationships()

        return self.schema_info

    def save_report(self, filename='database_schema_report.json'):
        """
        Save the comprehensive report to a JSON file
        
        :param filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(self.schema_info, f, indent=2, default=str)
        
        print(f"Comprehensive database schema report saved to {filename}")

def main():
    # Replace with your actual database connection string
    connection_string = 'postgresql://username:password@localhost:5432/your_database'
    
    try:
        # Initialize and analyze database
        analyzer = DatabaseSchemaAnalyzer(connection_string)
        
        # Generate comprehensive report
        comprehensive_report = analyzer.generate_comprehensive_report()
        
        # Pretty print key highlights
        print("\n--- Database Overview ---")
        print(json.dumps(comprehensive_report["database_metadata"], indent=2))
        
        print("\n--- Table Count and Details ---")
        print(f"Total Tables: {len(comprehensive_report['tables'])}")
        for table, details in comprehensive_report['tables'].items():
            print(f"\nTable: {table}")
            print(f"Columns: {len(details['columns'])}")
            print(f"Primary Keys: {details['primary_keys']}")
        
        # Save full report
        analyzer.save_report()
        
        print("\nDatabase schema analysis completed successfully!")
        print("Full report saved as 'database_schema_report.json'")
        print("Table relationships visualization saved as 'database_relationships.png'")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()