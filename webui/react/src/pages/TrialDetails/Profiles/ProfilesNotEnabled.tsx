import { Alert } from 'antd';
import React from 'react';

import Link from 'components/Link';
import { paths } from 'routes/utils';

const ProfilesNotEnabled: React.FC = () => {
  const description = (
    <>
      Learn about&nbsp;
      <Link
        external
        path={paths.docs('/training-apis/experiment-config.html#profiling')}
        popout>how to enable profiling on trials</Link>.
    </>
  );

  return (
    <Alert
      description={description}
      message="Profiling was not enabled for this trial."
      type="warning"
    />
  );
};

export default ProfilesNotEnabled;
