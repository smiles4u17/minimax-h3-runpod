import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import app
from job_list import sent_requests, merge_live_requests, sorted_requests


class JobListTests(unittest.TestCase):
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
