import unittest
from unittest import mock
import requests
import app
from runpod_monitor import endpoint_worker_logs, list_endpoint_workers, pod_logs, redact, safe_id, worker_logs


class MonitorTests(unittest.TestCase):
    def test_identifiers_and_log_redaction(self):
        with self.assertRaises(ValueError): safe_id('../settings')
        self.assertEqual(safe_id('job-id_u1'),'job-id_u1')
        result=redact('Bearer abc https://host/path?secret=xyz known-key',['known-key'])
        for secret in ('abc','xyz','known-key'): self.assertNotIn(secret,result)

    def test_sse_log_payload(self):
        r=mock.MagicMock()
        r.__enter__.return_value=r
        r.iter_lines.return_value=iter([b'id: 1',b'data: {"source":"container","line":"step 1","ts":"2026-09-20T00:00:00Z"}',b''])
        with mock.patch('runpod_monitor.requests.get',return_value=r):
            self.assertEqual(worker_logs('endpoint','worker','key')[0]['line'],'step 1')

    def test_pod_ndjson_log_payload(self):
        r=mock.MagicMock()
        r.__enter__.return_value=r
        r.iter_lines.return_value=iter([
            b'{"source":"system","line":"pulling image","ts":"2026-09-20T00:00:00Z"}',
            b'{"source":"container","line":"Bearer secret-key","ts":"2026-09-20T00:00:01Z"}',
        ])
        with mock.patch('runpod_monitor.requests.get',return_value=r) as get:
            result=pod_logs('pod-123','secret-key',source='both',tail=10)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['source'],'system')
        self.assertNotIn('secret-key',result[1]['line'])
        self.assertEqual({call.kwargs['params']['source'] for call in get.call_args_list}, {'container', 'system'})

    def test_pod_sse_log_payload(self):
        r=mock.MagicMock()
        r.__enter__.return_value=r
        r.iter_lines.return_value=iter([b'data: {"source":"container","line":"step 1"}',b''])
        with mock.patch('runpod_monitor.requests.get',return_value=r):
            self.assertEqual(pod_logs('pod-123','key')[0]['line'],'step 1')

    def test_endpoint_worker_ndjson_includes_worker_id(self):
        r=mock.MagicMock()
        r.__enter__.return_value=r
        r.iter_lines.return_value=iter([
            b'{"workerId":"worker-1","source":"container","line":"loading model","ts":"2026-09-20T00:00:00Z"}',
        ])
        with mock.patch('runpod_monitor.requests.get',return_value=r) as get:
            result=endpoint_worker_logs('endpoint','key',worker='worker-1',source='both',tail=10)
        self.assertEqual(result[0]['worker_id'],'worker-1')
        self.assertIn('/v2/serverless/endpoint/workers/worker-1/logs',get.call_args.args[0])
        self.assertEqual({call.kwargs['params']['source'] for call in get.call_args_list}, {'container', 'system'})

    def test_endpoint_worker_listing_accepts_items_envelope(self):
        r=mock.MagicMock()
        r.__enter__.return_value=r
        r.json.return_value={'items':[{'id':'worker-1'},{'workerId':'worker-2'}], 'summary':{}}
        with mock.patch('runpod_monitor.requests.get',return_value=r):
            self.assertEqual(list_endpoint_workers('endpoint','key'),['worker-1','worker-2'])

    def test_pod_route_keeps_configured_pod_and_returns_actual_lines(self):
        app.POD_LOG_CACHE.clear()
        settings={'runpod_api_key':'key','h3_storage':{'comfyui_pod_id':'pod-123'}}
        lines=[{'source':'system','line':'Pod started','ts':'2026-09-20T00:00:00Z'}]
        with (mock.patch.object(app,'settings',return_value=settings),
              mock.patch.object(app,'list_h3_comfyui_pods',return_value=[{'id':'pod-123'}]),
              mock.patch.object(app,'pod_logs',return_value=lines)):
            result=app.pod_worker_logs('pod-123')
        self.assertEqual(result['items'],lines)
        self.assertEqual(result['pod_id'],'pod-123')

    def test_endpoint_worker_route_returns_worker_ids(self):
        app.ENDPOINT_WORKER_LOG_CACHE.clear()
        settings={'runpod_api_key':'key','h3_endpoint_id':'endpoint'}
        lines=[{'source':'container','line':'handler started','ts':'2026-09-20T00:00:00Z','worker_id':'worker-1'}]
        with (mock.patch.object(app,'settings',return_value=settings),
              mock.patch.object(app,'list_endpoint_workers',return_value=['worker-1']),
              mock.patch.object(app,'endpoint_worker_logs',return_value=lines)):
            result=app.endpoint_workers_logs('endpoint')
        self.assertEqual(result['worker_ids'],['worker-1'])
        self.assertEqual(result['items'],lines)

    def test_cancel_terminal_job_does_not_mutate(self):
        with (mock.patch.object(app,'monitor_settings',return_value={'runpod_api_key':'key'}),
              mock.patch.object(app,'status',return_value={'status':'FAILED'}),
              mock.patch.object(app.requests,'post') as post):
            self.assertEqual(app.cancel_job('endpoint','job')['status'],'FAILED')
            post.assert_not_called()

    def test_cancel_running_job_records_exact_worker(self):
        with (mock.patch.object(app,'monitor_settings',return_value={'runpod_api_key':'key'}),
              mock.patch.object(app,'status',return_value={'status':'IN_PROGRESS','workerId':'worker'}),
              mock.patch.object(app.requests,'post') as post,
              mock.patch.object(app,'record_job_event') as record):
            app.cancel_job('endpoint','job')
            self.assertIn('/endpoint/cancel/job',post.call_args.args[0])
            self.assertEqual(record.call_args.args[0]['worker_id'],'worker')

    def test_h3_budget_reaches_runpod_policy(self):
        with mock.patch.object(app.requests,'post') as post:
            app.submit('endpoint','key',{'max_runtime_seconds':14400},{'timeout_seconds':3600})
            policy=post.call_args.kwargs['json']['policy']
            self.assertEqual(policy['executionTimeout'],14520000)
            self.assertGreaterEqual(policy['ttl'],86400000)

    def test_full_sampler_catalog_is_available(self):
        self.assertEqual(len(app.h3_sampling_options()['samplers']),46)
        for name in ('heun','dpmpp_sde_gpu','res_multistep_ancestral','uni_pc_bh2'):
            self.assertIn(name,app.H3_SAMPLERS)


if __name__=='__main__': unittest.main()
