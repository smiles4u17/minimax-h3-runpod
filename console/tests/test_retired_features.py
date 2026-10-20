import unittest
from pathlib import Path
from unittest import mock
import app


class RetirementTests(unittest.TestCase):
    def test_legacy_generation_routes_and_tabs_are_removed(self):
        routes = {getattr(route, 'path', '') for route in app.app.routes}
        self.assertNotIn('/api/run/infinitetalk', routes)
        self.assertNotIn('/api/run/wananimate', routes)
        html = (Path(__file__).parents[1] / 'static/index.html').read_text(encoding='utf-8')
        self.assertNotIn('data-tab="inf"', html)
        self.assertNotIn('data-tab="wan"', html)
        self.assertNotIn('samimateWanOnly', html)
        config = app.retire_legacy_settings({'infinite_endpoint_id': 'old', 'wananimate': {}, 'samimate': {'generation_backend': 'wan'}})
        self.assertNotIn('infinite_endpoint_id', config)
        self.assertNotIn('wananimate', config)
        self.assertEqual(config['samimate']['generation_backend'], 'h3')

    def test_retired_endpoint_history_does_not_poll_deleted_resources(self):
        with (mock.patch.object(app, 'settings', return_value={'h3_endpoint_id': 'h3', 'runpod_api_key': 'key'}),
              mock.patch.object(app, 'sent_requests', return_value={('up7pri63h6a97m', 'old'): {'status': 'IN_QUEUE'}, ('h3', 'new'): {'status': 'IN_QUEUE'}}),
              mock.patch.object(app.requests, 'get') as get):
            get.return_value.json.return_value = {'requests': []}
            result = app.get_sent_requests(force=True)
        self.assertEqual(get.call_count, 1)
        self.assertTrue(get.call_args.args[0].endswith('/h3/requests'))
        self.assertEqual(len(result['items']), 1)


if __name__ == '__main__':
    unittest.main()
