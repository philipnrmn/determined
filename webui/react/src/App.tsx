import { Button, notification } from 'antd';
import React, { useEffect, useLayoutEffect, useMemo, useState } from 'react';
import { HelmetProvider } from 'react-helmet-async';

import { setupAnalytics } from 'Analytics';
import Link from 'components/Link';
import Navigation from 'components/Navigation';
import PageMessage from 'components/PageMessage';
import Router from 'components/Router';
import Spinner from 'components/Spinner';
import StoreProvider, { StoreAction, useStore, useStoreDispatch } from 'contexts/Store';
import { useFetchInfo } from 'hooks/useFetch';
import useKeyTracker, { KeyCode, keyEmitter, KeyEvent } from 'hooks/useKeyTracker';
import usePolling from 'hooks/usePolling';
import useResize from 'hooks/useResize';
import useRouteTracker from 'hooks/useRouteTracker';
import useTheme from 'hooks/useTheme';
import Omnibar from 'omnibar/Omnibar';
import appRoutes from 'routes';
import { ThemeId } from 'themes';
import { BrandingType } from 'types';
import { correctViewportHeight, refreshPage } from 'utils/browser';

import css from './App.module.scss';
import { paths, serverAddress } from './routes/utils';

const AppView: React.FC = () => {
  const resize = useResize();
  const storeDispatch = useStoreDispatch();
  const { info, ui } = useStore();
  const [ canceler ] = useState(new AbortController());
  const { setThemeId } = useTheme();

  const isServerReachable = useMemo(() => !!info.clusterId, [ info.clusterId ]);

  const fetchInfo = useFetchInfo(canceler);

  useKeyTracker();
  useRouteTracker();

  // Poll every 10 minutes
  usePolling(fetchInfo, { interval: 600000 });

  useEffect(() => {
    setupAnalytics(info);

    /*
     * Check to make sure the WebUI version matches the platform version.
     * Skip this check for development version.
     */
    if (!process.env.IS_DEV && info.version !== process.env.VERSION) {
      const btn = <Button type="primary" onClick={refreshPage}>Update Now</Button>;
      const message = 'New WebUI Version';
      const description = <div>
        WebUI version <b>v{info.version}</b> is available.
        Check out what&apos;s new in our <Link external path={paths.docs('/release-notes.html')}>
          release notes
        </Link>.
      </div>;
      notification.warn({
        btn,
        description,
        duration: 0,
        key: 'version-mismatch',
        message,
        placement: 'bottomRight',
      });
    }
  }, [ info ]);

  // Detect branding changes and update theme accordingly.
  useEffect(() => {
    if (info.checked) {
      setThemeId(info.branding === BrandingType.HPE ? ThemeId.LightHPE : ThemeId.LightDetermined);
    }
  }, [ info.checked, info.branding, setThemeId ]);

  useEffect(() => {
    return () => canceler.abort();
  }, [ canceler ]);

  useEffect(() => {
    const keyDownListener = (e: KeyboardEvent) => {
      if (e.code === KeyCode.Space && e.ctrlKey) {
        if (ui.omnibar.isShowing) {
          storeDispatch({ type: StoreAction.HideOmnibar });
        } else {
          storeDispatch({ type: StoreAction.ShowOmnibar });
        }
      } else if (ui.omnibar.isShowing && e.code === KeyCode.Escape) {
        storeDispatch({ type: StoreAction.HideOmnibar });
      }
    };

    keyEmitter.on(KeyEvent.KeyDown, keyDownListener);

    return () => {
      keyEmitter.off(KeyEvent.KeyDown, keyDownListener);
    };
  }, [ ui.omnibar.isShowing, storeDispatch ]);

  // Correct the viewport height size when window resize occurs.
  useLayoutEffect(() => correctViewportHeight(), [ resize ]);

  if (!info.checked) {
    return <Spinner center />;
  }

  return (
    <div className={css.base}>
      {isServerReachable ? (
        <Navigation>
          <main>
            <Router routes={appRoutes} />
          </main>
        </Navigation>
      ) : (
        <PageMessage title="Server is Unreachable">
          <p>Unable to communicate with the server at &quot;{serverAddress()}&quot;.
              Please check the firewall and cluster settings.</p>
          <Button onClick={refreshPage}>Try Again</Button>
        </PageMessage>
      )}
      {ui.omnibar.isShowing && <Omnibar />}
    </div>
  );
};

const App: React.FC = () => {
  return (
    <HelmetProvider>
      <StoreProvider>
        <AppView />
      </StoreProvider>
    </HelmetProvider>
  );
};

export default App;
