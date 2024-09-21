class Broker:
    @staticmethod
    def log_event(event_message):
        """Loga um evento local."""
        print(f"Evento: {event_message}")
    
    @staticmethod
    def execute_command(command):
        """Executa um comando (por exemplo, notificar usuário)."""
        print(f"Comando executado: {command}")
