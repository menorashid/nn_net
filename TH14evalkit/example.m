clear all; close all; clc;
% running evaluation (simulated detection results)
% [pr_all,ap_all,map]=TH14evaldet('results/Run-2-det.txt','groundtruth','val');
[pr_all,ap_all,map]=TH14evaldet('results/Run-1-det_new.txt','groundtruth', 'test');

% plotting precision-recall results
overlapthresh=0.1;
ind=find([pr_all.overlapthresh]==overlapthresh);
clf
for i=1:length(ind)
  subplot(4,5,i)
  pr=pr_all(ind(i));
  plot(pr.rec,pr.prec)
  axis([0 1 0 1])
  title(sprintf('%s AP:%1.3f',pr.class,pr.ap))
end 