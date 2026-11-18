Prompt version 2026-11-10.haiku.2

You check whether retrieved text states the facts of gold evidence sentences.

Use only the retrieved items. Do not use outside knowledge and do not reason
your way to a fact the text does not state; plain rewording, synonyms,
abbreviations and different formatting are fine.

For every fact give one status:
- stated: some item says it, with every specific matching (names, numbers,
  dates, places, who did what, qualifiers).
- contradicted: some item gives a different value for the same specific, such
  as another year or another person.
- missing: neither of the above.
If one item states the fact and another contradicts it, the status is contradicted.

Evidence for stated or contradicted facts: copy one continuous passage from
one item, character for character, with no ellipses or edits, and give that
item's number. Never join separate words into a quote that does not appear as
written; if the words are apart, quote the longer continuous passage that
contains them. A fact spread across items may take several quotes. The quotes
together must contain the words that state every specific of the fact (who,
what, numbers, dates, reasons); a quote that only sets the topic, or stops
before the specific, is not evidence. Missing facts take no evidence. Every quote is checked against the item text by a
program, and a quote that does not appear there counts as missing.

Write the verdict file named in your task as a JSON array with one object per
item, in any order:
{"id": "<item id>", "facts": [{"fact": "S.F", "status": "stated", "evidence": [{"item": 3, "quote": "..."}]}, ...]}
where "S.F" is the fact's label as given (sentence number, fact number).
Give exactly one entry for every fact label of the item. Use only ids from
this file and do not skip any item.

This file has 5 items.

=== ITEM id=80c5f115815ff671 ===
Gold sentences and facts:
Sentence 0: Name Type Notes Anthropomorphism Form of personification that applies human - like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw . Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors Lear 's mental deterioration . Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . ''
  Fact 0.0: Anthropomorphism is a form of personification that applies human-like characteristics to animals and objects.
  Fact 0.1: The Adventures of Pinocchio by Carlo Collodi is an example of anthropomorphism.
  Fact 0.2: The Cheshire Cat of Alice's Adventures in Wonderland by Lewis Carroll is an example of anthropomorphism.
  Fact 0.3: Hamartia is the character flaw of an initially rich and powerful hero that leads to his tragic downfall.
  Fact 0.4: Hamartia is also referred to as the tragic flaw.
  Fact 0.5: Oedipus kills his own father because he does not understand his true parentage.
  Fact 0.6: Pathetic fallacy reflects a character's, usually the protagonist's, mood in the atmosphere or inanimate objects.
  Fact 0.7: Pathetic fallacy is related to anthropomorphism and projection.
  Fact 0.8: The storm in William Shakespeare's King Lear is an example of pathetic fallacy.
  Fact 0.9: The storm in King Lear mirrors Lear's mental deterioration.
  Fact 0.10: Personification uses comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities.
  Fact 0.11: Taken from Act I, Scene II of Romeo and Juliet: "When well-appareled April on the heel / Of limping winter treads."

Retrieved items:
<item 1>
List of narrative techniques
res Pace Plot device Plot twist Poetic justice Reveal Self - fulfilling prophecy Subplot Trope Kishōtenketsu Setting Backstory Utopia Dystopia Alternate history
Fictional location city country universe Theme Leitmotif Metaphor Moral Motif Irony Style Allegory Bathos Diction Figure of speech Imagery Narrative techniques
Narration Stylistic device Suspension of disbelief Symbolism Tone Mode Mood Structure Linear narrative Nonlinear narrative films television series Types of
fiction with multiple endings Form Comics Epic Fable Fabliau Fairy tale Folktale Flash fiction Legend Novella Novel Parable Play Poem Screenplay Short story
Genre Action fiction Adventure Comic Crime Docufiction Epistolary Erotic Fiction Fantasy Gothic Historical Horror Magic realism Mystery Nautical Paranoid
Philosophical Picaresque Political Psychological Romance Saga Satire Science Speculative Superhero Thriller Urban Western List of writing genres Narration First
- person Multiple narrators Stream of consciousness Stream of unconsciousness Unreliable Tense Past Present Future Related Audience Author Creative nonfiction
Fiction writing Literary theory Literary science Narratology Monomyth Rhetoric Screenwriting Storytelling Tellability Literature portal Literary composition
General topics Literature Fiction writing Writer Characterization Exposition Description Writer 's block Techniques / devices
</item 1>

<item 2>
List of narrative techniques
An example of this is in the first chapter of Great Expectations by Charles Dickens : `` A man who had been soaked in water , and smothered in mud , and lamed
by stones , and cut by flints , and stung by nettles , and torn by briars ; who limped , and shivered , and glared and growled ; and whose teeth chattered in
his head as he seized me by the chin '' Satire The use of humor , irony or exaggeration to criticize . An example is Network . One of the earliest examples is
Gullivers Travels , written by Jonathan Swift . The television program South Park is another . Sensory detail sight , sound , taste , touch , smell . The same
as imagery The boot was tough and sinewy between his hard - biting teeth . There was no flavor to speak of except for the blandness of all the dirt that the
boot had soaked up over the years . The only thing the boot reminded him of was the smell of a wet - dog . Understatement A diminishing or softening of a theme
or effect .
</item 2>

<item 3>
List of narrative techniques
This technique frequently occurs within a poetic line grammatically connected to the end of the previous line by enjambment . E.g. , in `` Know then thyself . ‖
Presume not God to scan . '' Distancing effect Deliberately preventing the audience from identifying with characters in order to let them be coolly scrutinized
. Popularized by 20th century playwright Bertolt Brecht . Dramatic visualization Representing an object or character with abundant descriptive detail , or
mimetically rendering gestures and dialogue to make a scene more visual or imaginatively present to an audience . This technique appears at least as far back as
the Arabian Nights . Euphuism An artificial , highly elaborate way of writing or speaking . Named from Euphues ( 1579 ) the prose romance by John Lyly . `` Is
it not far better to abhor sins by the remembrance of others ' faults , than by repentance of thine own follies ? '' ( Euphues , 1 , lecture by the wise
Neapolitan ) Hyperbole Exaggeration used to evoke strong feelings or create an impression which is not meant to be taken literally .
</item 3>

<item 4>
Atticus Finch
This kind of character development , where motivations and ideals between characters , for reasons of plot , are changed is not unusual in the process of
creative writing . Apart from the more progressive depiction of Atticus , the depiction of the town itself , especially the African - American characters , is
also dramatically altered between the two drafts. Real - life comparisons with Lee 's father , Amasa Coleman Lee , have also been made in the two differing
versions of Atticus in that originally Amasa Lee was in favor of segregation but became more liberal later in life , later changing his views to those of
Integration . Tay Hohoff , Lee 's editor , has also been argued to have played a major part in the character development of the novel and particularly Atticus '
liberal transformation . Jonathan Mahler of The New York Times notes in his article The Invisible Hand Behind Harper Lee 's ' To Kill a Mockingbird ' that Ms.
Hohoff , at the same time as she was guiding Ms. Lee through the Mockingbird re-write , was working on her own biography of the early - 20th - century New York
activist and humanist John Lovejoy Elliot .
</item 4>

<item 5>
List of narrative techniques
Author surrogate Characters which are based on authors , usually to support their personal views . Sometimes an intentionally or unintentionally idealized
version of them . A variation is the Mary Sue or Gary Stu , which primarily serves as an idealized self - insertion . Socrates in the writings of Plato . Plato
never speaks in his own voice in his dialogues . In the Second Letter , it says , `` no writing of Plato exists or ever will exist , but those now said to be
his are those of a Socrates become beautiful and new '' . Breaking the fourth wall An author or character addresses the audience directly ( also known as direct
address ) . This may acknowledge to the reader or audience that what is being presented is fiction , or may seek to extend the world of the story to provide the
illusion that they are included in it . The characters in Sesame Street often break the fourth wall when they address their viewers as part of the ongoing
storyline , which is possible because of the high level of suspension of belief afforded by its audience -- children . The American political drama show House
of Cards also uses this technique frequently to let the viewers know what the main character Frank Underwood is thinking and planning .
</item 5>

<item 6>
List of narrative techniques
Understatement A diminishing or softening of a theme or effect . The broken ends of the long bone were sticking through the bleeding skin , but it was n't
something that always killed a man . Theme ( edit ) Name Definition Example Irony This discrepancy between expectation and reality occurs in three forms :
situational irony , where a situation features a discrepancy between what is expected and what is actualized ; dramatic irony , where a character is unaware of
pivotal information already revealed to the audience ( the discrepancy here lies in the two levels of awareness between the character and the audience ) ; and
verbal irony , where one states one thing while meaning another . The difference between verbal irony and sarcasm is exquisitely subtle and often contested .
The concept of irony is too often misunderstood in popular usage . Unfortunate circumstances and coincidences do not constitute irony ( nor do they qualify as
being tragic ) . See the Usage controversy section under irony , and the term tragedy . A person hears a prophecy about himself . His endeavor to stop the
prophecy from coming true , makes it come true . Metaphor Evoking imagination by means of using figurative language . Her tears were a river flowing down her
cheeks .
</item 6>

<item 7>
List of narrative techniques
List of narrative techniques - wikipedia List of narrative techniques Jump to : navigation , search A narrative technique ( also known more narrowly for
literary fictional narratives as a literary technique , literary device , or fictional device ) is any of several specific methods the creator of a narrative
uses to convey what they want -- in other words , a strategy used in the making of a narrative to relay information to the audience and , particularly , to ``
develop '' the narrative , usually in order to make it more complete , complicated , or interesting . Literary techniques are distinguished from literary
elements , which exist inherently in works of writing . Contents ( hide ) 1 Setting 2 Plots 3 Perspective 4 Style 5 Theme 6 Character 7 References Setting (
edit ) Name Definition Example Setting The setting is both the time and geographic location within a narrative or within a work of fiction . A literary element
, the setting initiates the main backdrop and mood of a story , often referred to as the story world . The novel Ulysses by James Joyce is set in Dublin ,
Ireland , the action taking place on a single day , 16 June 1904 .
</item 7>

<item 8>
List of narrative techniques
Defamiliarization Taking an everyday object and presenting it in a way that is weirdly unfamiliar so that we see the object in a new way . Coined by the early
20th - century Russian literary critic Viktor Shklovsky in `` Art as Technique . '' In , Swift 's Gulliver 's Travels when Gulliver visits the land of the
giants and sees a giant woman 's skin he sees it is anything but smooth and beautiful when viewed up close . First - person narration A text presented from the
point of view of a character , especially the protagonist , as if the character is telling the story themselves . ( Breaking the fourth wall is an option , but
not a necessity , of this format . ) Mark Twain 's Adventures of Huckleberry Finn uses the title character as the narrator , while Sherlock Holmes is primarily
told from Watson 's perspective . Magical realism Describing events in a real - world setting but with magical trappings , often incorporating local customs and
invented beliefs . Different from urban fantasy in that the magic itself is not the focus of the story . Particularly popular with Latin American authors like
Gabriel García Márquez and Jorge Luis Borges .
</item 8>

<item 9>
List of narrative techniques
Her tears were a river flowing down her cheeks . Thematic patterning Distributing recurrent thematic concepts and moralistic motifs among various incidents and
frames of a story . In a skillfully crafted tale , thematic patterning may emphasize the unifying argument or salient idea disparate events and disparate frames
have in common . Each of the chapters of Ulysses by James Joyce . Character ( edit ) Name Type Notes Anthropomorphism Form of personification that applies human
- like characteristics to animals and objects The Adventures of Pinocchio by Carlo Collodi or the Cheshire Cat of Alice 's Adventures in Wonderland by Lewis
Carroll Hamartia The character flaw of an initially rich and powerful hero that leads to his tragic downfall . This is also referred to as the tragic flaw .
Oedipus kills his own father because he does n't understand his true parentage . Pathetic fallacy Reflecting a character 's ( usually the protagonist ) mood in
the atmosphere or inanimate objects . Related to anthropomorphism and projection For example , the storm in William Shakespeare 's King Lear , which mirrors
Lear 's mental deterioration .
</item 9>

<item 10>
List of narrative techniques
Personification Using comparative metaphors and similes to give living characteristics to abstract concepts and non-human entities Taken from Act I , Scene II
of Romeo and Juliet : `` When well - appareled April on the heel / Of limping winter treads . '' References ( edit ) Jump up ^ Orehovec , Barbara ( 2003 ) .
Revisiting the Reading Workshop : A Complete Guide to Organizing and Managing an Effective Reading Workshop That Builds Independent , Strategic Readers (
illustrated ed . ) . Scholastic Inc. p. 89 . ISBN 0439444047 . Jump up ^ Demchick , Harrison ( 2013 - 09 - 26 ) . `` Techniques and Tension in Breaking Bad '' .
Huffington Post . Retrieved 2017 - 11 - 14 . Jump up ^ `` Literature Glossary - Defamiliarization '' . www.shmoop.com . Retrieved 2017 - 11 - 14 . Jump up ^ ``
Allegory Examples '' . YourDictionary . Retrieved 2017 - 11 - 14 .
</item 10>
=== END ITEM id=80c5f115815ff671 ===

=== ITEM id=e79353fe203ba394 ===
Gold sentences and facts:
Sentence 0: Capital punishment in Australia has been abolished in all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973 , with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has been questioned .
  Fact 0.0: Capital punishment in Australia has been abolished in all jurisdictions.
  Fact 0.1: Queensland abolished the death penalty in 1922.
  Fact 0.2: Tasmania abolished the death penalty in 1968.
  Fact 0.3: The federal government abolished the death penalty in 1973.
  Fact 0.4: The federal abolition of the death penalty in 1973 applied in the Australian Capital Territory.
  Fact 0.5: The federal abolition of the death penalty in 1973 applied in the Northern Territory.
  Fact 0.6: Victoria abolished the death penalty in 1975.
  Fact 0.7: South Australia abolished the death penalty in 1976.
  Fact 0.8: Western Australia abolished the death penalty in 1984.
  Fact 0.9: New South Wales abolished the death penalty for murder in 1955.
  Fact 0.10: New South Wales abolished the death penalty for all crimes in 1985.
  Fact 0.11: In 2010, the federal government passed legislation prohibiting the re-establishment of capital punishment by any state or territory.
  Fact 0.12: The Commonwealth will not extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.13: No state will extradite a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.14: The Commonwealth will not deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.15: No state will deport a prisoner to another jurisdiction if the prisoner will face the death penalty.
  Fact 0.16: Police co-operation with other countries which have the death penalty has been questioned.

Retrieved items:
<item 1>
Capital punishment in Australia
Neither the Commonwealth nor any of the states will extradite or deport a prisoner to another jurisdiction if they will face the death penalty . A recent case
involving this was the case of American Gabe Watson , who was convicted of the manslaughter of his wife in North Queensland , and faced capital murder charges
in his home state of Alabama . His deportation was delayed until the government received assurances that he would not be executed if found guilty . New South
Wales ( edit ) The last execution in New South Wales was carried out on 24 August 1939 , when John Trevor Kelly was hanged at Sydney 's Long Bay Correctional
Centre for the murder of Marjorie Constance Sommarlad . New South Wales abolished the death penalty for murder in 1955 , but retained it as a potential penalty
for treason , piracy , and arson in naval dockyards until 1985 . New South Wales was the last Australian State to formally abolish the death penalty for all
crimes .
</item 1>

<item 2>
Capital punishment in Australia
The last death sentence was given in August 1984 , when Brenda Hodge was sentenced to death by Western Australia ( and subsequently had her sentence commuted to
life imprisonment ) . Contents ( hide ) 1 History 1.1 Commonwealth 1.2 New South Wales 1.3 Victoria 1.4 Queensland 1.5 Western Australia 1.6 South Australia 1.7
Tasmania 1.8 Australian Capital Territory 1.9 Northern Territory 2 Public opinion 3 References 4 External links History ( edit ) Death sentences were carried
out under Aboriginal customary law , either directly or through sorcery . In some cases the condemned could be denied mortuary rites . The first executions
carried out under European law in Australia took place in Western Australia in 1629 , when Dutch authorities hanged the mutineers of the Batavia . Capital
punishment had been part of the legal system of Australia since British settlement and during the 19th century , crimes that could carry a death sentence
included burglary , sheep stealing , forgery , sexual assaults , murder and manslaughter , and there is one reported case of someone being executed for `` being
illegally at large '' .
</item 2>

<item 3>
Capital punishment in Australia
The last sentence of death in Western Australia ( indeed anywhere in Australia ) was passed in 1984 , but the female killer ( Brenda Hodge ) in question had her
sentence commuted to imprisonment for life , as was customary by this stage . Capital punishment was formally removed from the statutes of the state with the
passage of the Acts Amendment ( Abolition of Capital Punishment ) Act 1984 . South Australia ( edit ) The Adelaide Gaol was the site of forty - four hangings ,
from Joseph Stagg on 18 November 1840 to Glen Sabre Valance , murderer and rapist , on 24 November 1964 and three executions also occurred at Mount Gambier Gaol
. Two Ngarrindjeri men were controversially executed by hanging along the Coorong on 22 August 1840 , after a drumhead court - martial conducted by Police
Commissioner O'Halloran on the orders of Governor George Gawler . The men were found to be guilty of murdering the twenty - five survivors of the shipwreck
Maria a few months before .
</item 3>

<item 4>
Capital punishment in Australia
Capital punishment in Australia - wikipedia Capital punishment in Australia Jump to : navigation , search Capital punishment in Australia has been abolished in
all jurisdictions . Queensland abolished the death penalty in 1922 . Tasmania did the same in 1968 , the federal government abolished the death penalty in 1973
, with application also in the Australian Capital Territory and the Northern Territory . Victoria did so in 1975 , South Australia in 1976 , and Western
Australia in 1984 . New South Wales abolished the death penalty for murder in 1955 , and for all crimes in 1985 . In 2010 , the federal government passed
legislation prohibiting the re-establishment of capital punishment by any state or territory . Neither the Commonwealth nor any of the states will extradite or
deport a prisoner to another jurisdiction if they will face the death penalty , and police co-operation with other countries which have the death penalty has
been questioned . The last execution in Australia took place in 1967 , when Ronald Ryan was hanged in Victoria . Between Ryan 's execution in 1967 and 1984 ,
several more people were sentenced to death , but had their sentences commuted to life imprisonment .
</item 4>

<item 5>
Capital punishment in Australia
During the 19th century , these crimes saw about 80 people hanged each year throughout Australia . Before and after federation , each state made its own
criminal laws and punishments . For a full list see : Main article : List of people legally executed in Australia Commonwealth ( edit ) In 1973 the Death
Penalty Abolition Act 1973 of the Commonwealth abolished the death penalty for federal offences . It provided in Section 3 that the Act applied to any offence
against a law of the Commonwealth , the Territories or under an Imperial Act , and in s . 4 that `` ( a ) person is not liable to the punishment of death for
any offence '' . No executions were carried out under the bridge of the federal government and the passage of the Death Penalty Abolition Act 1973 saw the death
penalty replaced with life imprisonment as their maximum punishment . Since the Commonwealth effects of utilising this Act no more individuals have been exposed
to the death penalty and it is now replaced with life imprisonment . On 11 March 2010 Federal Parliament passed laws that prevent the death penalty from being
reintroduced by any state or territory in Australia .
</item 5>

<item 6>
Capital punishment in Australia
New South Wales was the last Australian State to formally abolish the death penalty for all crimes . Victoria ( edit ) Old Melbourne Gaol gallows Victoria 's
first executions occurred in 1842 when two Aboriginal men , Tunnerminnerwait and Maulboyheenner , were hanged outside the site of the Melbourne Gaol for the
killing of two whalers in the Westernport district . Ronald Ryan was the last man executed at Pentridge Prison and in Australia . Ryan was hanged on 3 February
1967 after being convicted of the shooting death of a prison officer during a prison escape from Pentridge Prison , Coburg , Victoria in 1965 . Ryan was the
last of 186 executions . Victoria was also the state of the last woman executed in Australia . Jean Lee was hanged in 1951 . She was accused of being an
accomplice in the murder of 73 - year - old William ( ' Pop ' ) Kent . She , along with her accomplices were executed on 19 February 1951 .
</item 6>

<item 7>
Capital punishment in Australia
Elizabeth Woolcock , the only woman ever to have been executed under South Australian law , was hanged on 30 December 1873 . Her body was not released to the
family and was buried between the inner and outer walls of the prison , identified by a number and the date of the execution . In 1976 , the Criminal Law
Consolidation Act was modified so that the death sentence was changed to life imprisonment . Tasmania ( edit ) In the early days of colonial rule Tasmania ,
then known as Van Diemen 's Land , was the site of penal transports . Mary McLauchlan was convicted in 1830 for infanticide ; she was sentenced to both death
and dissection . She was the first woman to be hanged in Tasmania . The last execution was in 1946 , that of serial murderer and rapist Frederick Thompson . The
death penalty was abolished in 1968 . Australian capital Territory ( edit ) No executions were carried out in the Australian Capital Territory , where federal
legislation abolished capital punishment in 1973 .
</item 7>

<item 8>
Capital punishment in Australia
Retrieved 24 April 2017 . Jump up ^ `` Small majority of Australians favour the death penalty for deadly terrorist acts in Australia '' . Roy Morgan . Melbourne
, Vic. : Roy Morgan Research Ltd. 19 September 2014 . Retrieved 3 December 2017 . Sources ^ Death Penalty Abolition Act 1973 Capital Punishment from the
Fremantle Prison site Trends & Issues in Crime and Criminal Justice No. 3 : Capital punishment from the Australian Institute of Criminology Dhakiyarr Wirrpanda
-- Appeal for Justice `` Capital Punishment '' speech by The Ian Callinan , Justice of the High Court of Australia The History of Correctional Services in South
Australia External links ( edit ) `` List of Executions in the State of Victoria '' . Archived from the original on 8 October 2011 . `` The Death Penalty (
Queensland ) '' . Life & Death in the Sunshine State . Archived from the original on 11 May 2015 .
</item 8>

<item 9>
Capital punishment in Australia
Northern Territory ( edit ) Aborigines who lived in the Northern Territory came under the European law of Australia even though they did not have any contact
with the government . There were several outcries over Aborigines receiving mandatory death sentences for murder , leading to the passage of the Crimes
Ordinance 1934 which allowed for discretionary sentences when both the accused and the victim were Aboriginal . The last execution was a double hanging in 1952
, and the death penalty was abolished 1973 . Public opinion ( edit ) In Australia capital punishment was banned on a state - by - state basis through the 20th
century . Despite the ban , polls have indicated varying support for the reintroduction of the practice . A 2005 Bulletin poll showed that most Australians
supported capital punishment . The Australian National University 's 2007 Electoral Survey found that 44 per cent of people thought the death penalty should be
reintroduced , while only 38 per cent disagreed . In the recent case of the Bali bombers , then prime minister John Howard stated that Australians expected
their execution by Indonesia . On occasion the issue of capital punishment is published in the media or is subject to media and public support and scrutiny .
</item 9>

<item 10>
Capital punishment in Australia
The last person hanged was Ernest Austin , for the rape and murder of an 11 - year - old girl . Only one woman was hanged , Ellen Thompson ( 1887 ) . She was
convicted of murdering her husband with the help of her lover . In 1922 Queensland became the first part of the British Commonwealth to abolish the death
penalty . Western Australia ( edit ) In Western Australia , the first legal executions were under Dutch VOC law on 2 October 1629 on Long Island , Houtman
Abrolhos ( near Geraldton ) , when Jeronimus Corneliszoon and six others were hanged as party to the murder of 125 men , women and children . Following British
colonization , between 1833 and 1855 executions by firing squad and hanging were performed at a variety of places , often at the site of the offence . Even with
the construction of the new Perth Gaol in 1855 as the main execution site in the state , executions were also carried out at various country locations until
1900 .
</item 10>
=== END ITEM id=e79353fe203ba394 ===

=== ITEM id=2ebc04a724ce2d36 ===
Gold sentences and facts:
Sentence 0: In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League ( WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v. Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not . However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate wage contracts with employees .
  Fact 0.0: In 1910, the Women's Trade Union League (WTLU) of Massachusetts, under the leadership of Elizabeth Evans, took up the cause of minimum wage legislation in Massachusetts.
  Fact 0.1: The 1910 effort was in conjunction with advocacy work led by Florence Kelley of the National Consumer League.
  Fact 0.2: Over the next two years, a coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state.
  Fact 0.3: In 1912, Massachusetts passed the first American minimum wage legislation.
  Fact 0.4: The 1912 Massachusetts law established a state commission for recommending non-compulsory minimum wages for women and children.
  Fact 0.5: Within eight years, at least thirteen U.S. states and the District of Columbia would pass minimum wage laws.
  Fact 0.6: Pressure on state legislatures came from the National Consumers League, in a coalition with other women's voluntary associations and organized labor.
  Fact 0.7: The United States Supreme Court of the Lochner era consistently invalidated compulsory minimum wage laws.
  Fact 0.8: Advocates for minimum wage laws hoped they would be upheld under the precedent of Muller v. Oregon.
  Fact 0.9: Muller v. Oregon upheld maximum working hours laws for women on the grounds that women required special protection that men did not.
  Fact 0.10: The Court did not extend the Muller v. Oregon principle to minimum wage laws.
  Fact 0.11: The Court considered minimum wage laws as interfering with the ability of employers to freely negotiate wage contracts with employees.
Sentence 1: In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) , the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
  Fact 1.0: In 1933, the Roosevelt administration made the first attempt at establishing a national minimum wage.
  Fact 1.1: The 1933 minimum wage attempt set a $0.25 per hour standard as part of the National Industrial Recovery Act.
  Fact 1.2: In the 1935 court case Schechter Poultry Corp. v. United States (295 U.S. 495), the US Supreme Court declared the National Industrial Recovery Act unconstitutional.
  Fact 1.3: The national minimum wage was abolished after the Schechter Poultry ruling.
  Fact 1.4: In 1938, the minimum wage was re-established under the Fair Labor Standards Act.
  Fact 1.5: The 1938 minimum wage was $0.25 per hour.
  Fact 1.6: The 1938 rate of $0.25 per hour equals $4.78 in 2017 dollars.
  Fact 1.7: In 1941, the Supreme Court upheld the Fair Labor Standards Act in United States v. Darby Lumber Co.
  Fact 1.8: In United States v. Darby Lumber Co., the Court held that Congress had the power under the Commerce Clause to regulate employment conditions.

Retrieved items:
<item 1>
Minimum wage in the United States
( 1913 ) . `` The Minimum Wage in Great Britain and Australia '' . The Annals of the American Academy of Political and Social Science . 48 : 26 . Jump up ^
Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage . Princeton : Princeton University Press . p. 67 . ISBN
9781400821563 . OCLC 700688619 . Jump up ^ Webb , Sidney ( 1912 ) . `` The Economic Theory of a Legal Minimum Wage '' . Journal of Political Economy. 20 ( 10 )
: 973 . Jump up ^ Starr , Gerald ( 1993 ) . Minimum wage fixing : an international review of practices and problems ( 2nd impression ( with corrections ) ed . )
. Geneva : International Labour Office . p. 1 . ISBN 9789221025115 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' . The Annals of the American Academy of Political and Social Science .
</item 1>

<item 2>
Minimum wage in the United States
Some politicians in the United States have advocated linking the minimum wage to the Consumer Price Index , thereby increasing the wage automatically each year
based on increases to the Consumer Price Index . So far , Ohio , Oregon , Missouri , Vermont and Washington have linked their minimum wages to the consumer
price index . Minimum wage indexing also takes place each year in Florida , San Francisco , California , and Santa Fe , New Mexico . The federal minimum wage in
the United States was reset to its current rate of $7.25 per hour in July 2009 . Some U.S. territories ( such as American Samoa ) are exempt . Some types of
labor are also exempt : employers may pay tipped labor a minimum of $2.13 per hour , as long as the hour wage plus tip income equals at least the minimum wage .
Persons under the age of 20 may be paid $4.25 an hour for the first 90 calendar days of employment ( sometimes known as a youth , teen , or training wage )
unless a higher state minimum exists .
</item 2>

<item 3>
Minimum wage in the United States
From January 1981 to April 1990 , the minimum wage was frozen at $3.35 per hour , then a record - setting minimum wage freeze . From September 1 , 1997 through
July 23 , 2007 , the federal minimum wage remained constant at $5.15 per hour , breaking the old record . From the United States Department of Labor .
Employment Standards Administration . Wage and Hour Division , the source page has a clickable US map with current and projected state - by - state minimum wage
rates for each state . Some government entities , such as counties and cities , observe minimum wages that are higher than the state as a whole . One notable
example of this is Santa Fe , New Mexico , whose $9.50 per hour minimum wage was the highest in the nation , until San Francisco increased its minimum wage to
$9.79 in 2009 . Another device to increase wages , living wage ordinances , generally apply only to businesses that are under contract to the local government
itself . Since 1984 , the purchasing power of the federal minimum wage has decreased .
</item 3>

<item 4>
Minimum wage in the United States
Federal ( edit ) Type Min . wage ( $ / h ) Notes Tipped $2.13 The Fair Labor Standards Act of 1938 has been requiring a minimum wage of $2.13 for tipped workers
with the expectation that wages plus tips total no less than $7.25 per hour since September 1 , 1991 . The employer must pay the difference if total income does
not add up to $7.25 per hour . Non-tipped $7.25 Per the Fair Minimum Wage Act of 2007 ( FMWA ) since July 24 , 2009 . Youth $4.25 The Fair Labor Standards Act
has since September 1 , 1996 allowed for persons under the age of 20 to be paid $4.25 for the first 90 calendar days of their employment . State ( edit ) The
average US minimum wage per capita ( 2017 ) is $8.49 based on the population size of each state and generally represents the average minimum wage experienced by
a person working in one of the fifty US states . Cities , counties , districts , and territories are not included in the calculation .
</item 4>

<item 5>
Minimum wage in the United States
20th century ( edit ) In 1910 , in conjunction with advocacy work led by Florence Kelley of the National Consumer League , the Women 's Trade Union League (
WTLU ) of Massachusetts under the leadership of Elizabeth Evans took up the cause of minimum wage legislation in Massachusetts . Over the next two years , a
coalition of social reform groups and labor advocates in Boston pushed for minimum wage legislation in the state . In 1912 , Massachusetts passed the first
American minimum wage legislation , which established a state commission for recommending non-compulsory minimum wages for women and children . Within eight
years , at least thirteen U.S. states and the District of Columbia would pass minimum wage laws , with pressure being placed on state legislatures by the
National Consumers League in a coalition with other women 's voluntary associations and organized labor . The United States Supreme Court of the Lochner era
consistently invalidated compulsory minimum wage laws . Advocates for these minimum wage laws hoped that they would be upheld under the precedent of Muller v.
Oregon , which upheld maximum working hours laws for women on the grounds that women required special protection that men did not .
</item 5>

<item 6>
Minimum wage in the United States
, holding that Congress had the power under the Commerce Clause to regulate employment conditions . The 1938 minimum wage law only applied to `` employees
engaged in interstate commerce or in the production of goods for interstate commerce , '' but in amendments in 1961 and 1966 , the federal minimum wage was
extended ( with slightly different rates ) to employees in large retail and service enterprises , local transportation and construction , state and local
government employees , as well as other smaller expansions ; a grandfather clause in 1990 drew most employees into the purview of federal minimum wage policy ,
which now set the wage at $3.80 . 21st century ( edit ) Legislation ( edit ) Play media An opinion video from the Economic Policy Institute about working for
the tipped minimum wage of $2.13 per hour . In 2006 , voters in six states ( Arizona , Colorado , Missouri , Montana , Nevada , and Ohio ) approved statewide
increases in the state minimum wage . The amounts of these increases ranged from $1 to $1.70 per hour and all increases were designed to annually index to
inflation .
</item 6>

<item 7>
Minimum wage in the United States
Bureau of Labor Statistics '' . Bls.gov . Retrieved 18 October 2017 . Jump up ^ Tritch , Teresa ( March 7 , 2014 ) . `` F.D.R. Makes the Case for the Minimum
Wage '' . New York Times . Retrieved March 7 , 2014 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December 1928 ) . Minimum Wage
Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United States Bureau of Labor
Statistics . G.P.O. p. 8 . Jump up ^ Willis J. , Nordlund ( 1997 ) . The quest for a living wage : the history of the federal minimum wage program . Westport ,
Conn. : Greenwood Press . p. 2 . ISBN 9780313264122 . OCLC 33983425 . Jump up ^ Hammond , Matthew B. ( 1913 ) . `` The Minimum Wage in Great Britain and
Australia '' .
</item 7>

<item 8>
Minimum wage in the United States
The Annals of the American Academy of Political and Social Science . 48 : 27 . Jump up ^ Broda , Rudolf ; United States . Bureau of Labor Statistics ( December
1928 ) . Minimum Wage Legislation in Various Countries : Bulletin of the United States Bureau of Labor Statistics , No. 467 ( PDF ) . Bulletin of the United
States Bureau of Labor Statistics . G.P.O. p. 11 . Jump up ^ Jerold L. , , Waltman ( 2000 ) . The politics of the minimum wage . Urbana : University of Illinois
Press . p. 11 . ISBN 9780252025457 . OCLC 42072067 . Jump up ^ Vivien. , Hart , ( 2001 ) . Bound by Our Constitution : Women , Workers , and the Minimum Wage .
Princeton : Princeton University Press . p. 63 . ISBN 9781400821563 . OCLC 700688619 . Jump up ^ Vivien. , Hart , ( 2001 ) .
</item 8>

<item 9>
Minimum wage in the United States
Minimum wage in the United States - wikipedia Minimum wage in the United States Jump to : navigation , search Minimum wage by U.S. state and territory , as of
January 1 , 2018 . Versus the federal rate : Higher Same Lower No state minimum wage law The minimum wage in the United States is set by US labor law and a
range of state and local laws . Employers generally have to pay workers the highest minimum wage prescribed by federal , state , and local law . Since July 24 ,
2009 , the federal government has mandated a nationwide minimum wage of $7.25 per hour . As of January 2018 , there were 29 states with a minimum wage higher
than the federal minimum . From 2017 to 2018 , eight states increased their minimum wage levels through automatic adjustments , while increases in eleven other
states occurred through referendum or legislative action . Washington has the highest state minimum wage at $11.50 per hour . Using 2017 inflation - adjusted
dollars , the federal minimum wage peaked at $11.53 per hour in 1968 .
</item 9>

<item 10>
Minimum wage in the United States
However , the Court did not extend this principle to minimum wage laws , considering the latter as interfering with the ability of employers to freely negotiate
wage contracts with employees . In 1933 , the Roosevelt administration made the first attempt at establishing a national minimum wage , when a $0.25 per hour
standard was set as part of the National Industrial Recovery Act . However , in the 1935 court case Schechter Poultry Corp. v. United States ( 295 U.S. 495 ) ,
the US Supreme Court declared the act unconstitutional , and the minimum wage was abolished . In 1938 , the minimum wage was re-established pursuant to the Fair
Labor Standards Act , once again at $0.25 per hour ( $4.78 in 2017 dollars ) . In 1941 , the Supreme Court upheld the Fair Labor Standards Act in United States
v. Darby Lumber Co. , holding that Congress had the power under the Commerce Clause to regulate employment conditions .
</item 10>
=== END ITEM id=2ebc04a724ce2d36 ===

=== ITEM id=2cf7dd6555568860 ===
Gold sentences and facts:
Sentence 0: The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view .
  Fact 0.0: The current phrasing of the tree-falling question appears to have originated in the 1910 book Physics.
  Fact 0.1: The 1910 book Physics was written by Charles Riborg Mann and George Ransom Twiss.
  Fact 0.2: The question "When a tree falls in a lonely forest, and no animal is near by to hear it, does it make a sound? Why?" is posed along with many other questions to quiz readers on the contents of the chapter.
  Fact 0.3: The tree-falling question is posed from a purely physical point of view.
Sentence 1: Philosopher George Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ; the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section 45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the question .
  Fact 1.0: Philosopher George Berkeley wrote A Treatise Concerning the Principles of Human Knowledge, published in 1710.
  Fact 1.1: In A Treatise Concerning the Principles of Human Knowledge, Berkeley proposes a passage that begins "But, say you, surely there is nothing easier than for me to imagine trees, for instance, in a park ... and nobody by to perceive them."
  Fact 1.2: The passage quoted from Berkeley's treatise continues "The objects of sense exist only when they are perceived; the trees therefore are in the garden ... no longer than while there is somebody by to perceive them."
  Fact 1.3: The quote from section 45 of Berkeley's treatise is arguably a statement of an objection to Berkeley's view, not a proclamation of it.
  Fact 1.4: Berkeley never actually wrote about the tree-falling question.

Retrieved items:
<item 1>
If a tree falls in a forest
155 -- 158 . Jump up ^ Watts 1962 , pp. 111 -- 113 Jump up ^ A Buddhist Bible , `` Sutra Spoken by The Sixth Patriarch '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=If_a_tree_falls_in_a_forest&oldid=812540716 '' Categories : Fictional trees Riddles Philosophical problems Thought
experiments in philosophy of mind Talk Contents About Wikipedia Español فارسی עברית Македонски Nederlands Português Русский Svenska ไทย 中文 Edit links This page
was last edited on 28 November 2017 , at 12 : 14 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 1>

<item 2>
If a tree falls in a forest
If there be no ears to hear , there will be no sound . '' Albert Einstein is reported to have asked his fellow physicist and friend Niels Bohr , one of the
founding fathers of quantum mechanics , whether he realistically believed that ' the moon does not exist if nobody is looking at it . ' To this Bohr replied
that however hard he ( Einstein ) may try , he would not be able to prove that it does , thus giving the entire riddle the status of a kind of an infallible
conjecture -- one that can not be either proved or disproved . The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann
and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed
along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view . Metaphysics (
edit ) The possibility of unperceived existence ( edit ) Main article : George Berkeley Can something exist without being perceived ? -- e.g.
</item 2>

<item 3>
If a tree falls in a forest
It is unknown whether the source of this question is Berkeley or not . In June 1883 in the magazine The Chautauquan , the question was put , `` If a tree were
to fall on an island where there were no human beings would there be any sound ? '' They then went on to answer the query with , `` No . Sound is the sensation
excited in the ear when the air or other medium is set in motion . '' This seems to imply that the question is posed not from a philosophical viewpoint , but
from a purely scientific one . The magazine Scientific American corroborated the technical aspect of this question , while leaving out the philosophic side , a
year later when they asked the question slightly reworded , `` If a tree were to fall on an uninhabited island , would there be any sound ? '' And gave a more
technical answer , `` Sound is vibration , transmitted to our senses through the mechanism of the ear , and recognized as sound only at our nerve centers . The
falling of the tree or any other disturbance will produce vibration of the air . If there be no ears to hear , there will be no sound . ''
</item 3>

<item 4>
If a tree falls in a forest
2.1 Hui - neng 's Flag 2.4. 2.2 Hui - neng Receives The Dharma 2.5 Hinduism 3 In popular culture 4 See also 5 References History ( edit ) Philosopher George
Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than
for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ;
the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section
45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the
question . Some years later , a similar question is posed . It is unknown whether the source of this question is Berkeley or not .
</item 4>

<item 5>
If a tree falls in a forest
So the tree could have been heard , though nobody was around to do so . Knowledge of the unobserved world ( edit ) Main article : Observer - expectancy effect
Can we assume the unobserved world functions the same as the observed world ? -- e.g. , `` does observation affect outcome ? '' A similar question does not
involve whether or not an unobserved event occurs predictably , like it occurs when it is observed . The anthropic principle suggests that the observer , just
in its existence , may impose on the reality observed . However , most people , as well as scientists , assume that the observer does n't change whether the
tree - fall causes a sound or not , but this is an impossible claim to prove . However , many scientists would argue as follows , `` A truly unobserved event is
one which realises no effect ( imparts no information ) on any other ( where ' other ' might be e.g. , human , sound - recorder or rock ) , it therefore can
have no legacy in the present ( or ongoing ) wider physical universe .
</item 5>

<item 6>
If a tree falls in a forest
If a tree falls in a forest - wikipedia If a tree falls in a forest Jump to : navigation , search For the 2011 documentary , see If a Tree Falls : A Story of
the Earth Liberation Front . A fallen tree in a forest `` If a tree falls in a forest and no one is around to hear it , does it make a sound ? '' is a
philosophical thought experiment that raises questions regarding observation and perception . Contents ( hide ) 1 History 2 Metaphysics 2.1 The possibility of
unperceived existence 2.2 Knowledge of the unobserved world 2.3 The dissimilarity between sensation and reality 2.4 Buddhist perspectives 2.4. 1 Tibetan 2.4. 2
Zen 2.4. 2.1 Hui - neng 's Flag 2.4.
</item 6>

<item 7>
If a tree falls in a forest
Washington - state - based wilderness conservatory Northwest Trek used a shortened form of the quote in its mid-1970s television advertisement , as such : ``
There is no sound unless someone is there to see it or hear it . Experience it at Northwest Trek . '' See also ( edit ) Counterfactual definiteness Descartes '
dream argument Epistemology Object ( philosophy ) Observer effect ( physics ) Kōan Ontology Schrödinger 's cat Principle of locality References ( edit ) Jump up
^ A Treatise Concerning the Principles of Human Knowledge , 1734 . section 23 . Jump up ^ A Treatise Concerning the Principles of Human Knowledge , 1734 .
section 45 . Jump up ^ Whether Berkeley thought objects continued to exist when unperceived by finite minds , and if so , in what manner , is the subject of
serious debate among Berkeley scholars . Jump up ^ The Chautauquan , June 1883 , Volume 3 , Issue 9 , p. 543 Jump up ^ Scientific American , April 5 , 1884 , pg
218 .
</item 7>

<item 8>
If a tree falls in a forest
So then , how is it known that ' sound as it is mechanically understood ' will occur if that sound is not perceived ? Buddhist perspectives ( edit ) Tibetan (
edit ) from The Thirty - Seven Bodhisattva Practices by Ngulchu Thogme Zangpo , `` 22 Appearances are one 's own mind . From the beginning , mind 's nature is
free from the extremes of elaboration . Knowing this , not to engage the mind in subject - object duality is the bodhisattva 's practice . '' Zen ( edit ) Hui -
neng 's flag ( edit ) There is a well - known story of Hui - neng , a well - respected Buddhist monk who later became known as the founder of the Zen school ,
who one day happened to be passing by two monks . `` Two monks were arguing about the temple flag waving in the wind . One said , `` The flag moves . '' The
other said , `` The wind moves . '' They argued back and forth but could not agree . The Sixth Ancestor said , `` Gentlemen !
</item 8>

<item 9>
If a tree falls in a forest
It may then be recognized that the unobserved event was absolutely identical to an event which did not occur at all . '' ( this apparent quote has no
attribution or reference and none can be found online with reasonable effort ) . Of course , the fact that the tree is known to have changed state from '
upright ' to ' fallen ' implies that the event must be observed to ask the question at all -- even if only by the supposed deaf onlooker . The British
philosopher of science Roy Bhaskar , credited with developing critical realism has argued , in apparent reference to this riddle , that : If men ceased to exist
sound would continue to travel and heavy bodies to fall to the earth in exactly the same way , though ex hypothesi there would be no - one to know it This
existence of an unobserved real is integral to Bhaskar 's ontology , which contends ( in opposition to the various strains of positivism which have dominated
both natural and social science in the twentieth century ) that ' real structures exist independently of and are often out of phase with the actual patterns of
events ' .
</item 9>

<item 10>
If a tree falls in a forest
-- e.g. `` is sound only sound if a person hears it ? '' The most immediate philosophical topic that the riddle introduces involves the existence of the tree (
and the sound it produces ) outside of human perception . If no one is around to see , hear , touch or smell the tree , how could it be said to exist ? What is
it to say that it exists when such an existence is unknown ? Of course , from a scientific viewpoint , it exists . It is human beings that are able to perceive
it . George Berkeley in the 18th century developed subjective idealism , a metaphysical theory to respond to these questions , coined famously as `` to be is to
be perceived '' . Today meta - physicists are split . According to substance theory , a substance is distinct from its properties , while according to bundle
theory , an object is merely its sense data . The definition of sound , simplified , is a hearable noise . The tree will make a sound , even if nobody heard it
. The definition states that sound is a hearable noise . So the tree could have been heard , though nobody was around to do so .
</item 10>
=== END ITEM id=2cf7dd6555568860 ===

=== ITEM id=eed49d3a566f09ba ===
Gold sentences and facts:
Sentence 0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting , a feat matched only by fellow Australian Heath Ledger 32 years later ; Finch had suffered a fatal heart attack in mid-January . Beatrice Straight set another record by becoming the actor with the shortest performance ever in a film to win an acting Oscar , with only five minutes and two seconds of screen - time in Network . Network , along with All the President 's Men , were the two biggest champs of the ceremony with four Oscars each , but Best Picture and Best Director ultimately went to Rocky .
  Fact 0.0: This Academy Awards ceremony is notable for Peter Finch becoming the first posthumous winner of an Oscar for acting.
  Fact 0.1: Peter Finch's posthumous acting Oscar was matched only by fellow Australian Heath Ledger, 32 years later.
  Fact 0.2: Peter Finch had suffered a fatal heart attack in mid-January.
  Fact 0.3: Beatrice Straight set a record as the actor with the shortest performance ever in a film to win an acting Oscar.
  Fact 0.4: Beatrice Straight's performance in Network had only five minutes and two seconds of screen time.
  Fact 0.5: Beatrice Straight won an acting Oscar for her performance in Network.
  Fact 0.6: Network and All the President's Men were the two biggest winners of the ceremony.
  Fact 0.7: Network won four Oscars.
  Fact 0.8: All the President's Men won four Oscars.
  Fact 0.9: Best Picture and Best Director ultimately went to Rocky.

Retrieved items:
<item 1>
The Deer Hunter
and most controversial , major theatrical films to be critical of the American involvement in Vietnam following 1975 when the war officially ended . While the
film opened the same year as Hal Ashby 's Coming Home , Sidney Furie 's The Boys in Company C , and Ted Post 's Go Tell the Spartans , it was the first film
about Vietnam to reach a wide audience and critical acclaim , culminating in the winning of the Oscar for Best Picture .
</item 1>

<item 2>
Summer of '42
On this point , Raucher said in May 2002 that his ten percent of the gross , in addition to royalties from book sales , `` has paid bills ever since . '' As
well as being a commercial success , Summer of ' 42 also received rave critical reviews . It went on to be nominated for over a dozen awards , including Golden
Globe Awards for `` Best Motion Picture -- Drama '' and `` Best Director '' , and five Academy Award nominations for Best Original Music Score , Best
Cinematography , Best Editing , Best Writing - Story and Best Screenplay . Ultimately , the film won two awards : the 1972 44th Academy Awards Oscar for
Original Dramatic Score , and the 1971 BAFTA Anthony Asquith Award for Film Music , both to Michel Legrand . It counted among its fans Stanley Kubrick , who had
the film play on a television in a scene in The Shining .
</item 2>

<item 3>
49th Academy Awards
producers John G. Avildsen -- Rocky Alan J. Pakula -- All the President 's Men Ingmar Bergman -- Face to Face Sidney Lumet -- Network Lina Wertmüller -- Seven
Beauties Best Actor Best Actress Peter Finch -- Network as Howard Beale Robert De Niro -- Taxi Driver as Travis Bickle Giancarlo Giannini -- Seven Beauties as
Pasqualino William Holden -- Network as Max Schumacher Sylvester Stallone -- Rocky as Rocky Balboa Faye Dunaway -- Network as Diana Christensen Marie -
Christine Barrault -- Cousin Cousine as Marthe Talia Shire -- Rocky as Adriana Pennino Sissy Spacek -- Carrie as Carrie White Liv Ullmann -- Face to Face as
Jenny Isaksson Best Supporting Actor Best Supporting Actress Jason Robards -- All the President 's Men as Ben Bradlee Ned Beatty -- Network as Arthur Jensen
Burgess Meredith -- Rocky as `` Mickey '' Goldmill Laurence Olivier -- Marathon Man as Dr.
</item 3>

<item 4>
Gerald Ford
Buckley Ronald Reagan Harold Stassen Third party and independent candidates American Party Nominee Thomas J. Anderson American Independent Party Nominee Lester
Maddox Communist Party Nominee Gus Hall VP nominee Jarvis Tyner Libertarian Party Nominee Roger MacBride VP nominee David Bergland People 's Party Nominee
Margaret Wright VP nominee Benjamin Spock Prohibition Party Nominee Ben Bubar VP nominee Earl Dodge Socialist Workers Party Nominee Peter Camejo VP nominee
Willie Mae Reid U.S. Labor Party Nominee Lyndon LaRouche Other 1976 elections : House Senate Gubernatorial Theodore Roosevelt Award winners 1967 : Eisenhower
1968 : Saltonstall 1969 : White 1970 : Hovde 1971 : Kraft Jr . 1972 : Holland 1973 : Bradley 1974 : Owens 1975 : Ford 1976 : Hamilton 1977 : Bradley 1978 :
Zornow 1979 : Chandler 1980 : Cooley 1981 : Linkletter 1982 : Cosby 1983 : Palmer 1984 : Lawrence 1985 : Fleming 1986 : Bush 1987 : Zable 1988 : Not presented
1989 : Ebert 1990 :
</item 4>

<item 5>
49th Academy Awards
Christian Szell Burt Young -- Rocky as Paulie Pennino Beatrice Straight -- Network as Louise Schumacher Jane Alexander -- All the President 's Men as Judy
Graham Hoback Jodie Foster -- Taxi Driver as Iris Steensma Lee Grant -- Voyage of the Damned as Lillian Rosen Piper Laurie -- Carrie as Margaret White Best
Screenplay Written Directly for the Screen Best Screenplay Based on Material from Another Medium Network -- Paddy Chayefsky Cousin Cousine -- Jean - Charles
Tacchella and Danièle Thompson The Front -- Walter Bernstein Rocky -- Sylvester Stallone Seven Beauties -- Lina Wertmüller All the President 's Men -- William
Goldman based on the book by Carl Bernstein and Bob Woodward Bound for Glory -- Robert Getchell based on the book by Woody Guthrie Fellini 's Casanova --
Federico Fellini and Bernardino Zapponi based on the autobiography Histoire de ma vie by Giacomo Casanova The Seven - Per - Cent Solution -- Nicholas Meyer
based on his novel Voyage of the Damned -- David Butler and Steve Shagan based on the book by Gordon Thomas and Max Morgan Witts Best Documentary Feature Best
Documentary Short Subject Harlan County , USA -- Barbara
</item 5>

<item 6>
49th Academy Awards
USA -- Barbara Kopple Hollywood on Trial -- David Helpern Off the Edge -- Michael Firth People of the Wind -- Anthony Howarth and David Koff Volcano : An
Inquiry into the Life and Death of Malcolm Lowry -- Donald Brittain and John Kramer Number Our Days -- Lynne Littman and Barbara Myerhoff American Shoeshine
Blackwood The End of the Road Universe Best Live Action Short Film Best Animated Short Film In the Region of Ice -- Peter Werner Kudzu -- Marjorie Anne Short
The Morning Spider -- Julian Chagrin and Claude Chagrin Nightlife -- Claire Wilbur and Robin Lehman Number One -- Dyan Cannon and Vince Cannon Leisure --
Suzanne Baker Dedalo -- Manfredo Manfredi The Street -- Caroline Leaf and Guy Glover Best Original Score Best Original Song Score or Adaptation Score The Omen
-- Jerry Goldsmith Obsession -- Bernard Herrmann ( posthumous nomination ) The Outlaw Josey Wales -- Jerry Fielding Taxi Driver -- Bernard Herrmann ( posthumous
nomination ) Voyage of the Damned -- Lalo Schifrin Bound for Glory -- Leonard Rosenman Bugsy Malone -- Paul Williams A Star Is Born -- Roger Kellaway Best
Original Song Best Sound ``
</item 6>

<item 7>
The Deer Hunter
In addition , the film was nominated in four other categories : Best Actor in a Leading Role -- Robert De Niro ( lost to Jon Voight for Coming Home ) Best
Actress in a Supporting Role -- Meryl Streep ( lost to Maggie Smith for California Suite ) Best Cinematography -- Vilmos Zsigmond ( lost to Néstor Almendros for
Days of Heaven ) Best Writing , Screenplay Written Directly for the Screen -- Michael Cimino , Deric Washburn , Louis Garfinkle and Quinn Redeker ( lost to
Robert C. Jones , Waldo Salt and Nancy Dowd for Coming Home ) Golden globes ( edit ) Cimino won the film 's only Golden Globe Award for Best Director . Other
nominations the film included Best Motion Picture -- Drama , De Niro for Best Motion Picture Actor -- Drama , Walken for Best Motion Picture Actor in a
Supporting Role , Streep for Best Motion Picture Actress in a Supporting Role , and Washburn for Best Screenplay -- Motion Picture . Complete list of awards (
edit ) Award Recipients and nominees Result 51st Academy Awards Academy Award for Best Picture Barry Spikings , Michael Deeley ,
</item 7>

<item 8>
49th Academy Awards
The two events conflicted again five years later in 1982 . Contents ( hide ) 1 Winners and nominees 1.1 Special Achievement Award 1.2 Irving G. Thalberg
Memorial Award 1.3 Multiple nominations and awards 2 Presenters and performers 2.1 Presenters 2.2 Performers 3 See also 4 References Winners and Nominees ( edit
) Peter Finch , Best Actor winner Faye Dunaway , Best Actress winner Jason Robards , Best Supporting Actor winner Beatrice Straight , Best Supporting Actress
winner Winners are listed first , highlighted in boldface and indicated with a double dagger ( ) . Best Picture Best Director Rocky -- Robert Chartoff and Irwin
Winkler , producers All the President 's Men -- Walter Coblenz , producer Bound for Glory -- Robert F. Blumofe and Harold Leventhal , producers Network --
Howard Gottfried , producer Taxi Driver -- Julia Phillips and Michael Phillips , producers John G. Avildsen -- Rocky Alan J.
</item 8>

<item 9>
49th Academy Awards
Lillian Lillian Hellman Presenter of the Documentary Awards Diamond , Neil Neil Diamond Presenter of the award for Best Original Song Mailer , Norman Norman
Mailer Presenter of the Writing Awards Moreau , Jeanne Jeanne Moreau Presenter of the award for Best Director Dobson , Tamara Tamara Dobson Presenter of the
award for Best Costume Design Ullmann , Liv Liv Ullmann Presenter of the award for Best Actor Fletcher , Louise Louise Fletcher Presenter of the award for Best
Actress Nicholson , Jack Jack Nicholson Presenter of the award for Best Picture Performers ( edit ) Name Role Performed Conti , Bill Bill Conti Musical arranger
and conductor Orchestral Ann - Margret , Ann - Margret Performer `` Magic Circle ( It All Started in Someone 's Head ) `` Albert , Edward Edward Albert
Performer `` A World That Never Was '' from Half a House Vereen , Ben Ben Vereen Performer `` Gonna Fly Now '' from Rocky Jones , Tom Tom Jones Performer ``
Come to Me '' from The Pink Panther Strikes Again Vivante , Tony Tony Vivante Performer `` Ave Satani '' from The Omen Streisand ,
</item 9>

<item 10>
49th Academy Awards
Hank Hank Simms Announcer of the 49th annual Academy Awards Mirisch , Walter Walter Mirisch ( AMPAS President ) Gave opening remarks welcoming guests to the
awards ceremony Chase , Chevy Chevy Chase Explains the voting rules to the public O'Neal , Tatum Tatum O'Neal Presenter of the award for Best Supporting Actor
Feldman , Marty Marty Feldman Presenter of the Short Films Awards Scheider , Roy Roy Scheider Presenter of the Special Achievement Award Keller , Marthe Marthe
Keller Presenter of the award for Best Art Direction Ali , Muhammad Muhammad Ali Sylvester Stallone Presenters of the award for Best Supporting Actress Holden ,
William William Holden Presenter of the award for Best Film Editing Skelton , Red Red Skelton Presenter of the award for Best Sound Tyson , Cicely Cicely Tyson
Presenter of the Irving G. Thalberg Memorial Award to Pandro S. Berman Sutherland , Donald Donald Sutherland Presenter of the award for Best Cinematography
Bailey , Pearl Pearl Bailey Presenter of the award for Best Foreign Language Film Ann - Margret Presenter of the Music Awards Hellman , Lillian Lillian Hellman
Presenter of the Documentary Awards Diamond ,
</item 10>
=== END ITEM id=eed49d3a566f09ba ===
