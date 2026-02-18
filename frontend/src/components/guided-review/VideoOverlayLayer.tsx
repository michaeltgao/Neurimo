import { useEffect, useState, useCallback } from "react";
import type { RefObject } from "react";
import type { OverlayData, OverlayFrame } from "../../types/overlayTypes";
import { findFrameAtTime } from "../../types/overlayTypes";
import { getOverlayData } from "../../api/videos";
import SkeletonOverlay from "./SkeletonOverlay";
import HeadGazeOverlay from "./HeadGazeOverlay";
import ParentOverlay from "./ParentOverlay";
import EventCallout from "./EventCallout";

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  videoId: number;
  durationMs: number;
  currentTimeMs: number;
  showSkeleton: boolean;
  showHeadGaze: boolean;
  showParent?: boolean;
  showEventCallouts?: boolean;
};

export default function VideoOverlayLayer({
  videoRef,
  videoId,
  durationMs,
  currentTimeMs,
  showSkeleton,
  showHeadGaze,
  showParent = false,
  showEventCallouts = true,
}: Props) {
  const [overlayData, setOverlayData] = useState<OverlayData | null>(null);
  const [currentFrame, setCurrentFrame] = useState<OverlayFrame | null>(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch overlay data on mount
  useEffect(() => {
    let cancelled = false;

    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const data = await getOverlayData(videoId, durationMs);
        if (!cancelled) {
          setOverlayData(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to load overlay data");
          console.error("Failed to load overlay data:", err);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchData();
    return () => { cancelled = true; };
  }, [videoId, durationMs]);

  // Update video dimensions when video loads or resizes
  const updateDimensions = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;

    // Get the actual rendered size of the video
    const rect = video.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      setVideoDimensions({
        width: rect.width,
        height: rect.height,
      });
    }
  }, [videoRef]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    // Update dimensions when video metadata loads
    video.addEventListener("loadedmetadata", updateDimensions);
    video.addEventListener("resize", updateDimensions);

    // Initial update
    updateDimensions();

    // Also listen for window resize
    window.addEventListener("resize", updateDimensions);

    return () => {
      video.removeEventListener("loadedmetadata", updateDimensions);
      video.removeEventListener("resize", updateDimensions);
      window.removeEventListener("resize", updateDimensions);
    };
  }, [videoRef, updateDimensions]);

  // Find the current frame based on video time
  useEffect(() => {
    if (!overlayData?.frames || overlayData.frames.length === 0) {
      setCurrentFrame(null);
      return;
    }

    const tSec = currentTimeMs / 1000;
    const frame = findFrameAtTime(overlayData.frames, tSec);
    setCurrentFrame(frame);
  }, [overlayData, currentTimeMs]);

  // Don't render anything if overlays are disabled or no data
  if (!showSkeleton && !showHeadGaze && !showParent) return null;
  if (loading || error || !overlayData) return null;
  if (videoDimensions.width === 0 || videoDimensions.height === 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {/* Skeleton Overlay */}
      <SkeletonOverlay
        frame={currentFrame}
        videoWidth={videoDimensions.width}
        videoHeight={videoDimensions.height}
        visible={showSkeleton}
      />

      {/* Head/Gaze Overlay */}
      <HeadGazeOverlay
        frame={currentFrame}
        videoWidth={videoDimensions.width}
        videoHeight={videoDimensions.height}
        visible={showHeadGaze}
      />

      {/* Parent Overlay (bounding box + hands) */}
      <ParentOverlay
        frame={currentFrame}
        videoWidth={videoDimensions.width}
        videoHeight={videoDimensions.height}
        visible={showParent}
      />

      {/* Event Callout (shows current stimulus and response window) */}
      <EventCallout
        currentTimeSec={currentTimeMs / 1000}
        events={overlayData.events}
        expectedWindows={overlayData.expectedWindows}
        visible={showEventCallouts}
      />
    </div>
  );
}