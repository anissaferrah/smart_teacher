"""Smart Teacher — Code Executor Agent

Agent spécialisé pour l'exécution de code Python en sandbox sécurisé.
Utilise Docker pour l'isolation complète.
"""

import asyncio
import logging
import tempfile
import os
from typing import Dict, Any

log = logging.getLogger("SmartTeacher.CodeExecutor")

async def execute_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Exécute du code Python dans un conteneur Docker isolé.
    Retourne le résultat ou l'erreur.
    """
    try:
        # Vérifier que Docker est disponible
        import subprocess
        docker_check = await asyncio.create_subprocess_shell(
            "docker --version",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await docker_check.wait()
        if docker_check.returncode != 0:
            return {"success": False, "error": "Docker not available for secure code execution"}

        # Créer un fichier temporaire avec le code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Préparer la commande Docker sécurisée
            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # Pas d'accès réseau
                "--memory", "128m",   # Limite mémoire
                "--cpus", "0.5",      # Limite CPU
                "--read-only",        # Système de fichiers en lecture seule
                "--tmpfs", "/tmp",    # Seule /tmp est writable
                "--volume", f"{temp_file}:/code.py:ro",  # Monter le fichier en lecture seule
                "python:3.11-slim",   # Image légère Python
                "python", "/code.py"
            ]

            # Exécuter dans un conteneur Docker
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                if process.returncode == 0:
                    output = stdout.decode('utf-8', errors='replace').strip()
                    return {"success": True, "output": output}
                else:
                    error = stderr.decode('utf-8', errors='replace').strip()
                    return {"success": False, "error": error}

            except asyncio.TimeoutError:
                process.kill()
                return {"success": False, "error": f"Code execution timed out after {timeout}s"}

        finally:
            # Nettoyer le fichier temporaire
            try:
                os.unlink(temp_file)
            except:
                pass

    except Exception as e:
        log.error(f"Code execution error: {e}")
        return {"success": False, "error": f"Execution failed: {str(e)}"}

def execute_code_sync(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Version synchrone pour compatibilité.
    """
    import asyncio
    return asyncio.run(execute_code(code, timeout))
