import json
import unittest

from glide.dashboard_server import app


class DashboardPublicationResultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()

    def test_results_endpoint_summarizes_both_saved_matrices(self) -> None:
        response = self.client.get('/api/publication_matrix_results')

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(
            [item['scenario_count'] for item in payload['experiments']],
            [810, 162],
        )
        mixed = payload['experiments'][1]
        self.assertEqual(mixed['paired_trial_count'], 1620)
        self.assertEqual(mixed['scheduler_evaluations'], 4860)
        for metric, counts in mixed['hasp_vs_sjf'].items():
            self.assertEqual(
                counts['hasp_better'] + counts['tie'] + counts['hasp_worse'],
                162,
                metric,
            )

    def test_saved_matrix_reports_are_downloadable(self) -> None:
        for report_format, mime in (
            ('json', 'application/json'),
            ('csv', 'text/csv'),
            ('html', 'text/html'),
        ):
            with self.subTest(report_format=report_format):
                response = self.client.get(
                    f'/api/publication_matrix_report/mixed/{report_format}',
                )
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.mimetype.startswith(mime))
                self.assertGreater(len(response.data), 0)
                response.close()
        mixed_json = self.client.get('/api/publication_matrix_report/mixed/json')
        payload = json.loads(mixed_json.data)
        self.assertEqual(len(payload['results']['scenarios']), 162)
        mixed_json.close()

    def test_unknown_report_paths_are_rejected(self) -> None:
        response = self.client.get('/api/publication_matrix_report/../../etc/passwd')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()