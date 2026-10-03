"""Built-in Maintenance Status package entrypoint."""

from custom_components.mcp_assist.tools.packages.maintenance.maintenance import MaintenanceTool


class MaintenancePackageTool(MaintenanceTool):
    """Load the maintenance helper through the standard package API."""
