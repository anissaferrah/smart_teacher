#!/usr/bin/env python
"""
Réinitialiser la base de données — DROP tout, CREATE nouveau schéma.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from database.init_db import engine
from database.models import Base


async def reset_database():
    """DROP toutes les tables et recrée le schéma."""
    
    print("\n" + "="*70)
    print("🔄 RÉINITIALISATION DE LA BASE DE DONNÉES")
    print("="*70 + "\n")
    
    try:
        async with engine.begin() as conn:
            # DROP toutes les tables
            print("🗑️  Suppression des tables existantes...")
            await conn.run_sync(Base.metadata.drop_all)
            print("✅ Tables supprimées")
            
            # Créer toutes les tables
            print("🔨 Création des nouvelles tables...")
            await conn.run_sync(Base.metadata.create_all)
            print("✅ Tables créées avec le nouveau schéma")
        
        print("\n" + "="*70)
        print("✅ BASE DE DONNÉES RÉINITIALISÉE AVEC SUCCÈS!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    result = asyncio.run(reset_database())
    sys.exit(0 if result else 1)
