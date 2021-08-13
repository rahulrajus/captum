import React, { LegacyRef } from "react";

import "./LazyImage.css";
import cx from "../utils/cx";

interface ImageProps
  extends React.DetailedHTMLProps<
    React.ImgHTMLAttributes<HTMLImageElement>,
    HTMLImageElement
  > {
  srcProvider?: () => Promise<string>;
}

type LazyImageProps = ImageProps & IntersectionObserverInit;

/**
 * Wrapper over `<img/>` element to load images only when they are in view.
 * It uses IntersectionObserver to detect when the image is in view and then
 * sets the src attribute on `<img/>` which causes image to actually load.
 *
 * During the loading time if image goes out of view, the http request is
 * cancelled by setting src attribute back to "".
 *
 * Once the image is loaded, the observer disconnects and the images is never
 * changed, unless the props (src, root, threshold, rootMargin) are changed i.e.
 * component is re-rendered.
 *
 * This component also shows an overlay on the image area while the image
 * is loading. For best results, provide `width` and `height` props so that there
 * are no layout shifts once image loads.
 *
 * This component also provides an additional prop srcProvider which basically acts
 * like an async factory method which should return a base64 encoded string for
 * the image. This prop is invoked only once when the image in the view.
 * The facility of cancelling http calls is not available while using srcProvider
 * factory method.
 *
 * @param props LazyImageProps The `props` are combination of <img/> props and
 * IntersectionObserver config. Refer IntersectionObserver docs for understanding
 * root, threshold and rootMargin props. Remaining props are standard props on <img/>
 */
export function LazyImage(props: LazyImageProps) {
  const {
    src,
    srcProvider,
    root = null,
    threshold = [0, 1],
    rootMargin = "0px 0px 0px 0px",
    ...restImageParams
  } = props;

  const imgRef = React.useRef<HTMLImageElement>();
  const observer = React.useRef<IntersectionObserver>();
  const [loaded, setLoaded] = React.useState(false);
  const [finalSrc, setFinalSrc] = React.useState(src);

  React.useEffect(() => {
    if (!imgRef.current) {
      return;
    }

    observer.current?.disconnect();

    (observer as any).current = new IntersectionObserver(
      (entries) => {
        if (!imgRef.current) {
          return;
        }

        const imgEntry = entries.find(
          (entry) => entry.target === imgRef.current,
        );

        if (imgEntry?.intersectionRatio && imgEntry.intersectionRatio > 0) {
          if (!!src && imgRef.current.src !== src) {
            imgRef.current.src = src;
          } else if (src == null && !!srcProvider) {
            srcProvider().then((str) => {
              const dataUrl = `data:image/png;base64, ${str}`;
              if (imgRef.current != null) {
                imgRef.current.src = dataUrl;
              }
              setFinalSrc(dataUrl);
            });
          }
        } else {
          /*
           * Setting src to "" cancels the http request. This is necessary in the case
           * where the image comes into view and goes out before the image http call returns.
           */
          imgRef.current.src = "";
          setLoaded(false);
        }
      },
      { root, threshold, rootMargin },
    );

    observer.current?.observe(imgRef.current);

    const currObserver = observer.current;

    return () => currObserver?.disconnect();
  }, [src, root, rootMargin, threshold, srcProvider]);

  React.useEffect(() => {
    setLoaded(false);
  }, [src]);

  React.useEffect(() => {
    if (!!src) {
      setFinalSrc(src);
    }
  }, [src]);

  const handleImageLoad = (
    event: React.SyntheticEvent<HTMLImageElement, Event>,
  ) => {
    if (imgRef.current?.getAttribute("src") === finalSrc) {
      setLoaded(true);
      observer.current?.disconnect();
      props.onLoad && props.onLoad(event);
    }
  };

  const handleImageError = (
    event: React.SyntheticEvent<HTMLImageElement, Event>,
  ) => {
    if (imgRef.current?.getAttribute("src") === finalSrc) {
      setLoaded(true);
      observer.current?.disconnect();
      props.onError && props.onError(event);
    }
  };

  return (
    <div
      className="lazy-image-wrapper"
      style={{
        height: restImageParams.height,
        width: restImageParams.width,
      }}
    >
      <img
        {...restImageParams}
        ref={imgRef as LegacyRef<HTMLImageElement>}
        alt={props.alt}
        onLoad={handleImageLoad}
        onError={handleImageError}
      />
      <div className={cx(["lazy-image-loader", loaded ? "" : "visible"])} />
    </div>
  );
}
