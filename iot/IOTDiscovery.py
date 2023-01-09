from awscrt import io
from awsiot.greengrass_discovery import DiscoverResponse, DiscoveryClient
from concurrent import futures
from copy import copy
from IOTClient import IOTClient
from IOTContext import IOTContext, IOTCredentials
from typing import Optional


class IOTDiscovery:
    _discovery_client: DiscoveryClient
    _socket_options: io.SocketOptions
    _tls_context: io.ClientTlsContext
    _tls_options: io.ClientTlsContext
    context: IOTContext
    credentials: IOTCredentials

    def __init__(self, context: IOTContext, credentials: IOTCredentials):
        self.context = context
        self.credentials = credentials

        self._socket_options = io.SocketOptions()

        self._tls_options = io.TlsContextOptions.create_client_with_mtls_from_path(self.credentials.cert_path, self.credentials.priv_key_path)
        if (self.credentials.ca_path != None): self._tls_options.override_default_trust_store_from_path(None, self.credentials.ca_path)
        self._tls_context = io.ClientTlsContext(self._tls_options)

        self._discovery_client = DiscoveryClient(
            self.context.client_bootstrap,
            self._socket_options,
            self._tls_context,
            self.credentials.region,
        )

    def discover(self, thing: str) -> futures.Future:
        return self._discovery_client.discover(thing)

    def get_client(self, response: DiscoverResponse) -> Optional[IOTClient]:
        for group in response.gg_groups:
            for core in group.cores:
                for conn in core.connectivity:
                    try:
                        print(f"Trying core {core.thing_arn} at host {conn.host_address} port {conn.port}")


                        credentials = copy(self.credentials)
                        credentials.endpoint = conn.host_address
                        credentials.port = conn.port

                        client = IOTClient(self.context, credentials, group.certificate_authorities[0].encode("utf-8"))
                        client.connect().result()
                        print("Connected!")
                        return client

                    except Exception as e:
                        print(f"Connection failed with exception {e}")
                        continue
