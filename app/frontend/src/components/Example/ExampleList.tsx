import {Example} from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {text: "What is the weather today?", value: "What is the weather today?"},
    {text: "How can I send my son an email?", value: "How can I send my son an email?"},
    {text: "Tell me about the musical 'The Phantom of the Opera'", value: "Tell me about the musical 'The Phantom of the Opera'"}
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({onExampleClicked}: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked}/>
                </li>
            ))}
        </ul>
    );
};
