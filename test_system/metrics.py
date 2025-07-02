class Metrics:
    def __init__(self):
        self.processing_count = 0
        self.total_processing_time = 0
        self.total_processed_faces = 0
        self.processing_time = {
            'face_detection': 0,
            'face_tracking': 0,
            'roi_selection': 0,
            'signal_extraction': 0,
            'signal_extraction_preprocess': 0,
            'signal_extraction_inference': 0,
            'hr_extraction': 0,
        }

    def fill_average(self):
        all_frames_metrics = set(['face_detection', 'face_tracking', 'roi_selection'])
        self.avg_processing_time = {}
        for metric_name, total_time in self.processing_time.items():
            if metric_name in all_frames_metrics:
                avg_time = (total_time / self.processing_count * 1000) if self.processing_count > 0 else 0
                self.avg_processing_time[metric_name] = avg_time
            else: # divide by total processed faces for metrics not in all frames
                avg_time = (total_time / self.total_processed_faces * 1000) if self.total_processed_faces > 0 else 0
                self.avg_processing_time[metric_name] = avg_time

        self.avg_total_processing_time = (self.total_processing_time / self.processing_count * 1000) if self.processing_count > 0 else 0

    def __str__(self):
        # Create table headers
        table = "Metrics Summary\n"
        table += "=" * 100 + "\n"
        table += f"{'Metric':<35} {'Total Time (s)':<15} {'Average Time (ms)':<18} {'Count':<10}\n"
        table += "-" * 100 + "\n"
        
        # Add processing count row
        table += f"{'Processing Count':<35} {'-':<15} {'-':<18} {self.processing_count:<10}\n"
        
        # Add total processing time row
        avg_total = self.avg_total_processing_time
        table += f"{'Total Processing':<35} {self.total_processing_time:<15.4f} {avg_total:<18.2f} {'-':<10}\n"
        
        # Add individual processing times
        for metric_name, total_time in self.processing_time.items():
            avg_time = (self.avg_processing_time[metric_name] if metric_name in self.avg_processing_time else 0)
            display_name = metric_name.replace('_', ' ').title()
            table += f"{display_name:<35} {total_time:<15.4f} {avg_time:<18.2f} {'-':<10}\n"
        
        table += "=" * 100
        return table
