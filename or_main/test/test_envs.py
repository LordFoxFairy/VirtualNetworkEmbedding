import unittest
import networkx as nx

from or_main.environments.vne_env import VNEEnvironment


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = VNEEnvironment(None)

    def test_env(self):
        print(len(list(nx.connected_components(self.env.SUBSTRATE.net))))

        num_disconnected_vnrs = 0
        for vnr in self.env.VNRs_INFO.values():
            if len(list(nx.connected_components(vnr.net))) > 1:
                num_disconnected_vnrs += 1
                #print(len(list(nx.connected_components(vnr.net))))
        print(num_disconnected_vnrs, len(self.env.VNRs_INFO))


if __name__ == '__main__':
    unittest.main()