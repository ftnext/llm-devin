import click
import llm


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("name")
    def devin(name):
        "Act as Devin AI"
        click.echo(f"Hello from Devin AI to {name}!")
