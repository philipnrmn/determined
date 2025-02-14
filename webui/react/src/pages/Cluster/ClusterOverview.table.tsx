import { ColumnType } from 'antd/es/table';

import { ResourcePool } from 'types';
import { alphanumericSorter, numericSorter } from 'utils/sort';
import { V1ResourcePoolTypeToLabel } from 'utils/types';

export const columns: ColumnType<ResourcePool>[] = [
  {
    dataIndex: 'name',
    key: 'name',
    sorter: (a: ResourcePool, b: ResourcePool): number => alphanumericSorter(a.name, b.name),
    title: 'Pool Name',
  },
  {
    dataIndex: 'description',
    key: 'description',
    sorter: (a: ResourcePool, b: ResourcePool): number =>
      alphanumericSorter(a.description, b.description),
    title: 'Description',
  },
  {
    key: 'chart',
    title: 'GPU Slots Allocation',
  },
  {
    dataIndex: 'type',
    key: 'type',
    render: (_, record) => V1ResourcePoolTypeToLabel[record.type],
    sorter: (a: ResourcePool, b: ResourcePool): number => alphanumericSorter(a.type, b.type),
    title: 'Type',
  },
  {
    dataIndex: 'numAgents',
    key: 'numAgents',
    sorter: (a: ResourcePool, b: ResourcePool): number =>
      numericSorter(a.numAgents, b.numAgents),
    title: 'Agents',
  },
  {
    dataIndex: 'slotsAvailable',
    key: 'slotsAvailable',
    sorter: (a: ResourcePool, b: ResourcePool): number =>
      numericSorter(a.slotsAvailable, b.slotsAvailable),
    title: 'Total Slots',
  },
  {
    dataIndex: 'auxContainerCapacity',
    key: 'auxContainerCapacity',
    sorter: (a: ResourcePool, b: ResourcePool): number =>
      numericSorter(a.auxContainerCapacity, b.auxContainerCapacity),
    title: 'Max Aux Containers Per Agent',
  },
  {
    dataIndex: 'auxContainersRunning',
    key: 'auxContainersRunning',
    sorter: (a: ResourcePool, b: ResourcePool): number =>
      numericSorter(a.auxContainersRunning, b.auxContainersRunning),
    title: 'CPUs Used',
  },
];
