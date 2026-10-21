import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import app
from job_list import sent_requests, merge_live_requests, sorted_requests


class JobListTests(unittest.TestCase):
    def test_expired_bulk_status_recovers_verified_completion_without_downloading(self):
        app.REQUEST_RECOVERY_CACHE.clear()
        row = {'endpoint_id': 'ep', 'job_id': 'expired', 'status': 'SUBMITTED'}
        response = mock.Mock()
        response.json.return_value = {'requests': []}
        with (mock.patch.object(app, 'settings', return_value={'h3_endpoint_id': 'ep', 'runpod_api_key': 'key'}),
              mock.patch.object(app, 'sent_requests', side_effect=lambda *a: {('ep', 'expired'): dict(row)}),
              mock.patch.object(app.requests, 'get', return_value=response),
              mock.patch.object(app, 'recover_completed_h3_status', return_value={'status': 'COMPLETED'}) as recover,
              mock.patch.object(app, 'read_job_diagnostics', return_value={'elapsed_seconds': 1414, 'worker_id': 'gpu-worker', 'updated_at': '2026-10-21T20:09:16Z'}),
              mock.patch.object(app, 'save_outputs') as download):
            for _ in range(2):
                result = app.get_sent_requests(force=True)['items'][0]
                self.assertEqual(result['status'], 'COMPLETED')
                self.assertEqual(result['execution_ms'], 1414000)
                self.assertEqual(result['status_source'], 'verified_volume_output')
                self.assertFalse(result['can_cancel'])
            recover.assert_called_once()
            download.assert_not_called()
        app.REQUEST_RECOVERY_CACHE.clear()

    def test_absent_or_unverified_completion_stays_unavailable(self):
        app.REQUEST_RECOVERY_CACHE.clear()
        response = mock.Mock()
        response.json.return_value = {'requests': []}
        with (mock.patch.object(app, 'settings', return_value={'h3_endpoint_id': 'ep', 'runpod_api_key': 'key'}),
              mock.patch.object(app, 'sent_requests', return_value={('ep', 'missing'): {'endpoint_id': 'ep', 'job_id': 'missing', 'status': 'SUBMITTED'}}),
              mock.patch.object(app.requests, 'get', return_value=response),
              mock.patch.object(app, 'recover_completed_h3_status', return_value=None)):
            self.assertEqual(app.get_sent_requests(force=True)['items'][0]['status'], 'UNAVAILABLE')
        app.REQUEST_RECOVERY_CACHE.clear()

    def test_worker_error_is_failed_and_queue_duration_is_retained(self):
        rows = {}
        merge_live_requests(rows, [None, {'id': 'failed', 'status': 'COMPLETED',
                            'output': {'error': 'CUDA failure'}, 'delayTime': 9000}], 'ep')
        self.assertEqual(rows[('ep', 'failed')]['status'], 'FAILED')
        self.assertEqual(rows[('ep', 'failed')]['queue_ms'], 9000)

    def test_live_queue_includes_unrecorded_jobs_and_disables_missing_requests(self):
        rows = {('ep', 'old'): {'endpoint_id': 'ep', 'job_id': 'old', 'status': 'SUBMITTED'},
                ('ep', 'done'): {'endpoint_id': 'ep', 'job_id': 'done', 'status': 'COMPLETED'}}
        merge_live_requests(rows, [{'id': 'new', 'status': 'IN_QUEUE'}], 'ep')
        self.assertEqual(rows[('ep', 'old')]['status'], 'UNAVAILABLE')
        self.assertEqual(rows[('ep', 'done')]['status'], 'COMPLETED')
        self.assertTrue(rows[('ep', 'new')]['live'])
        merge_live_requests(rows, [{'id':'done','status':'IN_PROGRESS'},
                                  {'id':'delivered','status':'IN_PROGRESS','output':{'stage':'completed'}}], 'ep')
        self.assertEqual(rows[('ep','done')]['status'], 'COMPLETED')
        self.assertEqual(rows[('ep','delivered')]['status'], 'FINALIZING')

    def test_receipts_preserve_submissions_after_history_rolls_over(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'new_receipt.json').write_text(json.dumps({'endpoint_id':'ep','job':{'id':'new','status':'IN_QUEUE'},'created_at':'2026-10-10T09:00:00+00:00','generation':{'seed':12}}))
            (root/'new_request.json').write_text('must not be read as metadata')
            (root/'new.txt').write_text('Status: SUBMITTED\nGeneration request:\n'+json.dumps({'input':{'prompt':'Full saved prompt\nSecond line','model':'model.safetensors','steps':8}}))
            (root/'bad_receipt.json').write_text('invalid JSON')
            history = [{'endpoint_id':'ep','job_id':'old','status':'COMPLETED','target':'status'},
                       {'endpoint_id':'ep','job_id':'old','status':'SUBMITTED','target':'h3','time':'2026-10-09 22:00:00','debug':{'seed':9}}]
            rows = sent_requests(history, root)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[('ep','old')]['status'], 'COMPLETED')
            self.assertEqual(rows[('ep','old')]['target'], 'h3')
            self.assertEqual(rows[('ep','new')]['seed'], 12)
            self.assertEqual(rows[('ep','new')]['prompt'], 'Full saved prompt\nSecond line')
            self.assertEqual(sorted_requests(rows)[0]['job_id'], 'new')

    def test_list_uses_bulk_metadata_and_never_downloads_outputs(self):
        response=mock.Mock()
        response.json.return_value={'requests':[{'id':'running','status':'IN_PROGRESS'},{'id':'done','status':'COMPLETED'}]}
        with (mock.patch.object(app,'settings',return_value={'h3_endpoint_id':'ep','runpod_api_key':'key'}),
              mock.patch.object(app,'sent_requests',return_value={}),
              mock.patch.object(app.requests,'get',return_value=response) as get,
              mock.patch.object(app,'save_outputs') as download):
            result=app.get_sent_requests(force=True)
        self.assertEqual(get.call_count, 1)
        self.assertTrue(get.call_args.args[0].endswith('/ep/requests'))
        self.assertEqual({x['job_id']:x['can_cancel'] for x in result['items']},{'running':True,'done':False})
        download.assert_not_called()
