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

This file has 6 items.

=== ITEM id=de7d06a9c6f89ed3 ===
Gold sentences and facts:
Sentence 0: The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet Bandits and must outrun and out - prank them again .
  Fact 0.0: Kevin and his family decide to take a trip to Florida.
  Fact 0.1: Kevin takes the wrong plane.
  Fact 0.2: Kevin ends up in New York City.
  Fact 0.3: Kevin tries to make do with what he has.
  Fact 0.4: Kevin uses his father's credit card to stay at the Plaza Hotel.
  Fact 0.5: Kevin is soon confronted by the Wet Bandits.
  Fact 0.6: Kevin must outrun and out-prank the Wet Bandits again.

Retrieved items:
<item 1>
Home Alone 2: Lost in New York
Disc One Home Alone ( Main Title ) ( 2 : 07 ) This Year 's Wish ( 1 : 47 ) We Overslept Again / Holiday Flight ( 3 : 19 ) Separate Vacations * ( 1 : 58 )
Arrival in New York * * ( 2 : 59 ) The Thieves Return ( 3 : 28 ) Plaza Hotel ( 3 : 04 ) Concierge ( 1 : 31 ) Distant Goodnights ( Christmas Star ) ( Lyrics by
Leslie Bricusse ) ( 2 : 05 ) A Day in the City ( : 59 ) Duncan 's Toy Store ( 2 : 41 ) Turtle Doves ( 1 : 29 ) To the Plaza , Presto ( 3 : 27 ) Race to the Room
/ Hot Pursuit ( 4 : 08 ) Haunted Brownstone ( 3 : 02 ) Appearance of the Pigeon Lady ( 3 : 21 ) Christmas at Carnegie Hall ( 5 : 15 ) O Come ,
</item 1>

<item 2>
Home Alone 2: Lost in New York
Contents ( hide ) 1 Plot 2 Cast 2.1 Main cast 2.2 The McCallister family 2.3 Other characters 3 Release 3.1 Marketing 3.2 Box office 3.3 Critical reception 3.4
Home media 3.5 Social Media 4 Music 4.1 Original Score 4.2 Original Soundtrack Album 4.3 The Deluxe Edition 5 Novelization 6 Sequels 7 See also 8 References 9
External links Plot ( edit ) In Chicago , the McCallister family is preparing for a Christmas vacation in Miami . On the night before their departure , the
entire family gathers at Peter and Kate 's home , where their youngest son , Kevin , sees Florida as contradictory to Christmas , on his opinion of the lack of
Christmas trees in Florida . During the school Christmas pageant , Kevin 's older brother Buzz humiliates him during his solo , causing Kevin to retaliate ,
thereby ruining the performance .
</item 2>

<item 3>
Home Alone 2: Lost in New York
While Buzz fakes an apology to Kevin in front of the McCallisters ( who are convinced it is sincere ) , Kevin refuses to apologize to Buzz and storms to the
third floor of the house . Prior to the pageant , Peter has unknowingly caused the alarm clock to reset ; consequently , the family once again oversleeps . In
the confusion they have to rush to reach the airport on time , Kevin falls behind everyone else , and mistakenly boards a flight bound for New York City while
trying to replace the batteries for a Talkboy , carrying Peter 's bag containing his wallet and a large amount of money ; upon arrival in Miami , Kate realizes
that Kevin is missing again . In New York , Kevin tours the city and meets a homeless woman tending to pigeons at Central Park , where he runs off . Kevin goes
to the Plaza Hotel , where he tantalizes the staff into checking in with Peter 's credit card . On Christmas Eve , Kevin tours the city in a limousine and
visits a toy store where he meets its philanthropic owner , Mr. Duncan .
</item 3>

<item 4>
Home Alone 2: Lost in New York
Duncan . Kevin learns that the proceeds from the store 's Christmas sales will be donated to a children 's hospital . Kevin donates $20 to the hospital . As a
token of appreciation , Mr. Duncan offers Kevin a pair of ceramic turtledoves as a gift , instructing him to give one to another person as a sign of eternal
friendship . After running into Harry and Marv , who have recently escaped from prison and are now called `` The Sticky Bandits '' , Kevin retreats to the Plaza
Hotel . The hotel 's concierge , Mr. Hector , confronts Kevin about the credit card , which has been reported stolen . Kevin flees after evading Mr. Hector ,
but is captured by Harry and Marv . The duo discuss plans for breaking into the toy store that night . Kevin manages to escape by pinching a woman in front of
him , which results in the woman attacking the duo . Kevin 's family travels to New York after tracking the whereabouts of the stolen credit card , and Kate
searches the city for Kevin .
</item 4>

<item 5>
Home Alone 2: Lost in New York
When the duo chase Kevin around the townhouse , he escapes and calls the police . Harry and Marv catch him and take him down to Central Park to kill him , but
the pigeon lady sneaks in and throws birdseed on the duo , and the pigeons storm in to incapacitate them . Kevin sets off fireworks he had bought earlier to
signal the police . The police arrive and arrest the bandits . At the toy store , Mr. Duncan finds a note from Kevin attached to the brick , explaining about
what happened . Kate , still searching for Kevin on foot , remembers his fondness for Christmas trees . After observing Kevin making a wish at the Rockefeller
Center Christmas Tree , Kate meets him there , and they reconcile . On Christmas Day , a truckload of gifts arrive at the McCallisters ' hotel room from the toy
store . Kevin and Buzz reconcile , and Buzz allows him to open up the first present . Kevin goes to Central Park to give the pigeon lady the second turtledove .
At the Plaza , Buzz receives the bill for Kevin 's stay from Cedric and shows it to Peter .
</item 5>

<item 6>
Home Alone 2: Lost in New York
In the beginning of the novelization , a prologue , which ends up being Marv 's nightmare in prison , he and Harry sneak away from the cops and return to Kevin
's house to seek revenge on Kevin . Kevin bolts into the garage with Marv and Harry in hot pursuit . Harry and Marv end up triggering extra traps that Kevin had
set up in the garage . Kevin watches as Marv ends up triggering a trap where a running lawnmower falls on his head ( This was a trap featured in Home Alone 3 )
. Sequels ( edit ) Main article : Home Alone 3 A second sequel , Home Alone 3 , followed in 1997 . Two additional television movie sequels , Taking Back the
House and The Holiday Heist , would follow in 2002 and 2012 , respectively . See also ( edit ) List of films featuring fictional films References ( edit ) Jump
up ^ `` HOME ALONE 2 - LOST IN NEW YORK ( PG ) '' . British Board of Film Classification . November 4 , 2017 . Retrieved November 10 , 2015 .
</item 6>

<item 7>
Home Alone 2: Lost in New York
Meanwhile , Kevin goes to his uncle Rob 's townhouse , only to find the house vacant and undergoing renovations , while Rob and his family are in Paris . At
Central Park , he encounters and befriends the pigeon lady . They go to Carnegie Hall , where an orchestra is performing . The pigeon lady explains how her life
collapsed and how she dealt with it by taking care of the pigeons in the park . Kevin gives the pigeon lady some advice and promises that he will be her friend
. Kevin , after remembering what the bandits said , returns to the townhouse , and rigs it with numerous booby traps . Kevin arrives at the toy store during
Harry and Marv 's robbery and throws a brick through the window , setting off the store 's alarm . Kevin then lures the duo to the townhouse , where he springs
the traps and Harry and Marv suffer various injuries . When the duo chase Kevin around the townhouse , he escapes and calls the police .
</item 7>

<item 8>
Home Alone 2: Lost in New York
It is the second film in the Home Alone series and the sequel to Home Alone . Macaulay Culkin reprises his role as Kevin McCallister , while Joe Pesci and
Daniel Stern reprise their roles as the Wet Bandits , now known as the Sticky Bandits . Catherine O'Hara , John Heard , Rob Schneider , Tim Curry , and Brenda
Fricker are also featured . The plot is as follows : Kevin and his family decide to take a trip to Florida , but Kevin takes the wrong plane and ends up in New
York City . He tries to make do with what he has , such as using his father 's credit card to stay at the Plaza Hotel , but is soon confronted by the Wet
Bandits and must outrun and out - prank them again . The film was shot in Winnetka , Illinois ; O'Hare International Airport in Chicago ; Evanston , Illinois ;
and New York City ( which was star Culkin 's hometown at the time ) . The exterior of Duncan 's Toy Chest in New York City was filmed outside of the Rookery
Building in downtown Chicago .
</item 8>

<item 9>
Home Alone 2: Lost in New York
Peter suddenly calls out , `` Kevin , you spent $967 on room service ? ! '' at which point Kevin runs off . Cast ( edit ) Main cast ( edit ) Macaulay Culkin as
Kevin McCallister Joe Pesci as Harry Lime Daniel Stern as Marv Merchants The McCallister family ( edit ) Catherine O'Hara as Kate McCallister John Heard as
Peter McCallister Devin Ratray as Buzz McCallister , Kevin 's brother who gets him into trouble Kieran Culkin ( Macaulay Culkin 's younger brother ) as Fuller
McCallister Gerry Bamman as Frank McCallister , Kevin 's uncle Hillary Wolf as Megan McCallister Maureen Elisabeth Shay ( who replaces Angela Goethals ) as
Linnie McCallister Mike Maronna as Jeff McCallister Terrie Snell as Leslie McCallister , Kevin 's aunt Jedidiah Cohen as Rod McCallister Senta Moses as Tracy
McCallister Daiana Campeanu as Sondra McCallister Anna Slotky as Brooke McCallister Other characters ( edit ) Brenda Fricker as the Pigeon Lady Tim Curry as Mr.
</item 9>

<item 10>
Home Alone 2: Lost in New York
External links ( edit ) Wikiquote has quotations related to : Home Alone 2 : Lost in New York Home Alone 2 : Lost in New York on IMDb Home Alone 2 : Lost in New
York at the TCM Movie Database Home Alone 2 : Lost in New York at AllMovie Home Alone 2 : Lost in New York at Rotten Tomatoes Movie Locations Guide.com - Maps
and directions to Home Alone 2 Filming Locations Home Alone Films Home Alone ( 1990 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone 3 ( 1997 ) Home Alone
4 ( 2002 ) Home Alone : The Holiday Heist ( 2012 ) Video games Home Alone ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Home Alone ( 2006 ) Miscellaneous
Home Alone : Original Motion Picture Soundtrack Talkboy Chris Columbus Directed Adventures in Babysitting ( 1987 ) Heartbreak Hotel ( 1988 ) Home Alone ( 1990 )
Only the Lonely ( 1991 ) Home Alone 2 : Lost in New York ( 1992 ) Mrs.
</item 10>
=== END ITEM id=de7d06a9c6f89ed3 ===

=== ITEM id=f6f971f314683a0a ===
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
So the tree could have been heard , though nobody was around to do so . Knowledge of the unobserved world ( edit ) Main article : Observer - expectancy effect
Can we assume the unobserved world functions the same as the observed world ? -- e.g. , `` does observation affect outcome ? '' A similar question does not
involve whether or not an unobserved event occurs predictably , like it occurs when it is observed . The anthropic principle suggests that the observer , just
in its existence , may impose on the reality observed . However , most people , as well as scientists , assume that the observer does n't change whether the
tree - fall causes a sound or not , but this is an impossible claim to prove . However , many scientists would argue as follows , `` A truly unobserved event is
one which realises no effect ( imparts no information ) on any other ( where ' other ' might be e.g. , human , sound - recorder or rock ) , it therefore can
have no legacy in the present ( or ongoing ) wider physical universe .
</item 1>

<item 2>
If a tree falls in a forest
-- e.g. `` is sound only sound if a person hears it ? '' The most immediate philosophical topic that the riddle introduces involves the existence of the tree (
and the sound it produces ) outside of human perception . If no one is around to see , hear , touch or smell the tree , how could it be said to exist ? What is
it to say that it exists when such an existence is unknown ? Of course , from a scientific viewpoint , it exists . It is human beings that are able to perceive
it . George Berkeley in the 18th century developed subjective idealism , a metaphysical theory to respond to these questions , coined famously as `` to be is to
be perceived '' . Today meta - physicists are split . According to substance theory , a substance is distinct from its properties , while according to bundle
theory , an object is merely its sense data . The definition of sound , simplified , is a hearable noise . The tree will make a sound , even if nobody heard it
. The definition states that sound is a hearable noise . So the tree could have been heard , though nobody was around to do so .
</item 2>

<item 3>
If a tree falls in a forest
2.1 Hui - neng 's Flag 2.4. 2.2 Hui - neng Receives The Dharma 2.5 Hinduism 3 In popular culture 4 See also 5 References History ( edit ) Philosopher George
Berkeley , in his work , A Treatise Concerning the Principles of Human Knowledge ( 1710 ) , proposes , `` But , say you , surely there is nothing easier than
for me to imagine trees , for instance , in a park ( ... ) and nobody by to perceive them . ( ... ) The objects of sense exist only when they are perceived ;
the trees therefore are in the garden ( ... ) no longer than while there is somebody by to perceive them . '' ( It is worth noting that the quote from section
45 is arguably a statement of an objection to Berkeley 's view , and not a proclamation of it . ) Nevertheless , Berkeley never actually wrote about the
question . Some years later , a similar question is posed . It is unknown whether the source of this question is Berkeley or not .
</item 3>

<item 4>
If a tree falls in a forest
It is unknown whether the source of this question is Berkeley or not . In June 1883 in the magazine The Chautauquan , the question was put , `` If a tree were
to fall on an island where there were no human beings would there be any sound ? '' They then went on to answer the query with , `` No . Sound is the sensation
excited in the ear when the air or other medium is set in motion . '' This seems to imply that the question is posed not from a philosophical viewpoint , but
from a purely scientific one . The magazine Scientific American corroborated the technical aspect of this question , while leaving out the philosophic side , a
year later when they asked the question slightly reworded , `` If a tree were to fall on an uninhabited island , would there be any sound ? '' And gave a more
technical answer , `` Sound is vibration , transmitted to our senses through the mechanism of the ear , and recognized as sound only at our nerve centers . The
falling of the tree or any other disturbance will produce vibration of the air . If there be no ears to hear , there will be no sound . ''
</item 4>

<item 5>
If a tree falls in a forest
If a tree falls in a forest - wikipedia If a tree falls in a forest Jump to : navigation , search For the 2011 documentary , see If a Tree Falls : A Story of
the Earth Liberation Front . A fallen tree in a forest `` If a tree falls in a forest and no one is around to hear it , does it make a sound ? '' is a
philosophical thought experiment that raises questions regarding observation and perception . Contents ( hide ) 1 History 2 Metaphysics 2.1 The possibility of
unperceived existence 2.2 Knowledge of the unobserved world 2.3 The dissimilarity between sensation and reality 2.4 Buddhist perspectives 2.4. 1 Tibetan 2.4. 2
Zen 2.4. 2.1 Hui - neng 's Flag 2.4.
</item 5>

<item 6>
If a tree falls in a forest
It may then be recognized that the unobserved event was absolutely identical to an event which did not occur at all . '' ( this apparent quote has no
attribution or reference and none can be found online with reasonable effort ) . Of course , the fact that the tree is known to have changed state from '
upright ' to ' fallen ' implies that the event must be observed to ask the question at all -- even if only by the supposed deaf onlooker . The British
philosopher of science Roy Bhaskar , credited with developing critical realism has argued , in apparent reference to this riddle , that : If men ceased to exist
sound would continue to travel and heavy bodies to fall to the earth in exactly the same way , though ex hypothesi there would be no - one to know it This
existence of an unobserved real is integral to Bhaskar 's ontology , which contends ( in opposition to the various strains of positivism which have dominated
both natural and social science in the twentieth century ) that ' real structures exist independently of and are often out of phase with the actual patterns of
events ' .
</item 6>

<item 7>
If a tree falls in a forest
155 -- 158 . Jump up ^ Watts 1962 , pp. 111 -- 113 Jump up ^ A Buddhist Bible , `` Sutra Spoken by The Sixth Patriarch '' Retrieved from ``
https://en.wikipedia.org/w/index.php?title=If_a_tree_falls_in_a_forest&oldid=812540716 '' Categories : Fictional trees Riddles Philosophical problems Thought
experiments in philosophy of mind Talk Contents About Wikipedia Español فارسی עברית Македонски Nederlands Português Русский Svenska ไทย 中文 Edit links This page
was last edited on 28 November 2017 , at 12 : 14 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply .
By using this site , you agree to the Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit
organization . About Wikipedia
</item 7>

<item 8>
If a tree falls in a forest
`` The mind is like akasa ( space ) . Just as there are the objects in the akasa , so there are thoughts in the mind . The akasa is the counterpart of the mind
and objects are of thought . One can not hope to measure the universe and study the phenomena . It is impossible . For the objects are mental creations ... The
universe is only an object created by the mind and has its being in the mind . '' ( Talks with Sri Ramana Maharshi , pp. 478 -- 479 ) . `` The shrutis ( a
subdivision of the Vedas ) and the sages say that the objects are only mental creations . They have no substantive being . Investigate the matter and ascertain
the truth of the statement . The result will be the conclusion that the objective world is in the subjective consciousness . '' ( TWSRM , p. 479 ) . In popular
culture ( edit ) Canadian songwriter , social activist and environmentalist Bruce Cockburn poses the question in the chorus of his 1989 song `` If a Tree Falls
'' and frames it with his lyrics as a pressing question regarding the cause and effect of deforestation .
</item 8>

<item 9>
If a tree falls in a forest
Washington - state - based wilderness conservatory Northwest Trek used a shortened form of the quote in its mid-1970s television advertisement , as such : ``
There is no sound unless someone is there to see it or hear it . Experience it at Northwest Trek . '' See also ( edit ) Counterfactual definiteness Descartes '
dream argument Epistemology Object ( philosophy ) Observer effect ( physics ) Kōan Ontology Schrödinger 's cat Principle of locality References ( edit ) Jump up
^ A Treatise Concerning the Principles of Human Knowledge , 1734 . section 23 . Jump up ^ A Treatise Concerning the Principles of Human Knowledge , 1734 .
section 45 . Jump up ^ Whether Berkeley thought objects continued to exist when unperceived by finite minds , and if so , in what manner , is the subject of
serious debate among Berkeley scholars . Jump up ^ The Chautauquan , June 1883 , Volume 3 , Issue 9 , p. 543 Jump up ^ Scientific American , April 5 , 1884 , pg
218 .
</item 9>

<item 10>
If a tree falls in a forest
If there be no ears to hear , there will be no sound . '' Albert Einstein is reported to have asked his fellow physicist and friend Niels Bohr , one of the
founding fathers of quantum mechanics , whether he realistically believed that ' the moon does not exist if nobody is looking at it . ' To this Bohr replied
that however hard he ( Einstein ) may try , he would not be able to prove that it does , thus giving the entire riddle the status of a kind of an infallible
conjecture -- one that can not be either proved or disproved . The current phrasing appears to have originated in the 1910 book Physics by Charles Riborg Mann
and George Ransom Twiss . The question `` When a tree falls in a lonely forest , and no animal is near by to hear it , does it make a sound ? Why ? '' is posed
along with many other questions to quiz readers on the contents of the chapter , and as such , is posed from a purely physical point of view . Metaphysics (
edit ) The possibility of unperceived existence ( edit ) Main article : George Berkeley Can something exist without being perceived ? -- e.g.
</item 10>
=== END ITEM id=f6f971f314683a0a ===

=== ITEM id=55dab9f2700570fa ===
Gold sentences and facts:
Sentence 0: The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly .
  Fact 0.0: The Roosevelt Corollary was an addition to the Monroe Doctrine.
  Fact 0.1: President Theodore Roosevelt articulated the Roosevelt Corollary.
  Fact 0.2: Theodore Roosevelt articulated the Roosevelt Corollary in his State of the Union address in 1904.
  Fact 0.3: The Roosevelt Corollary was articulated after the Venezuela Crisis of 1902-03.
  Fact 0.4: The Roosevelt Corollary states that the United States will intervene in conflicts between European countries and Latin American countries.
  Fact 0.5: Under the Roosevelt Corollary, the United States intervenes to enforce legitimate claims of the European powers.
  Fact 0.6: Under the Roosevelt Corollary, the United States intervenes rather than having the Europeans press their claims directly.

Retrieved items:
<item 1>
Truman Doctrine
Long - term policy and metaphor ( edit ) The Truman Doctrine underpinned American Cold War policy in Europe and around the world . In the words of historian
James T. Patterson , `` The Truman Doctrine was a highly publicized commitment of a sort the administration had not previously undertaken . Its sweeping
rhetoric , promising that the United States should aid all ' free people ' being subjugated , set the stage for innumerable later ventures that led to
globalistic commitments . It was in these ways a major step . '' The doctrine endured , historian Dennis Merill argues , because it addressed a broader cultural
insecurity regarding modern life in a globalized world . It dealt with Washington 's concern over communism 's domino effect , it enabled a media - sensitive
presentation of the doctrine that won bipartisan support , and it mobilized American economic power to modernize and stabilize unstable regions without direct
military intervention . It brought nation - building activities and modernization programs to the forefront of foreign policy . The Truman Doctrine became a
metaphor for emergency aid to keep a nation from communist influence .
</item 1>

<item 2>
Truman Doctrine
George F. Kennan ( 1904 - 2005 ) proposed the doctrine of containment in 1946 . In February 1946 , Kennan , an American diplomat in Moscow , sent his famed ``
Long Telegram '' , which predicted the Soviets would only respond to force and that the best way to handle them would be through a long - term strategy of
containment , that is stopping their geographical expansion . After the British warned that they could no longer help Greece , and following Prime Minister
Konstantinos Tsaldaris 's visit to Washington in December 1946 to ask for American assistance , the U.S. State Department formulated a plan . Aid would be given
to both Greece and Turkey , to help cool the long - standing rivalry between them . American policy makers recognized the instability of the region , fearing
that if Greece was lost to communism , Turkey would not last long . Similarly , if Turkey yielded to Soviet demands , the position of Greece would be endangered
. A regional domino effect threat therefore guided the American decision .
</item 2>

<item 3>
Roosevelt Corollary
The Roosevelt Corollary was an addition to the Monroe Doctrine articulated by President Theodore Roosevelt in his State of the Union address in 1904 after the
Venezuela Crisis of 1902 -- 03 . The corollary states that the United States will intervene in conflicts between European countries and Latin American countries
to enforce legitimate claims of the European powers , rather than having the Europeans press their claims directly . Roosevelt tied his policy to the Monroe
Doctrine , and it was also consistent with his foreign policy included in his Big Stick Diplomacy . Roosevelt stated that in keeping with the Monroe Doctrine ,
the United States was justified in exercising `` international police power '' to put an end to chronic unrest or wrongdoing in the Western Hemisphere . While
the Monroe Doctrine had sought to prevent European intervention , the Roosevelt Corollary was used to justify US intervention throughout the hemisphere . In
1934 , President Franklin D. Roosevelt renounced interventionism and established his Good Neighbor policy for the Western Hemisphere .
</item 3>

<item 4>
Roosevelt Corollary
of Nations European Union Latin America United Nations Third Border Initiative International organizations ( hide ) Doctrines , policies , concepts Presidential
doctrines Proclamation of Neutrality Monroe Roosevelt Corollary Good Neighbor policy Truman Eisenhower Kennedy Johnson Nixon Carter Reagan Clinton Bush Obama
Other doctrines Lodge Corollary Stimson Kirkpatrick Weinberger Powell Rumsfeld Wolfowitz Policies and concepts Blowback Containment Domino theory
Non-interventionism Progressive realism Rollback Special Relationship Taiwan Relations Act United States intervention in Latin America Policy Monroe Doctrine (
1823 ) Platt Amendment ( 1901 -- 04 ) Roosevelt Corollary / Big Stick ideology ( 1904 ) Good Neighbor policy ( 1933 ) Dollar diplomacy Banana Wars Wars Mexican
-- American War ( 1846 - 48 ) Spanish -- American War ( 1898 ) Border War ( 1910 -- 19 ) United States involvement in the Mexican Revolution ( 1916 -- 19 )
Overt actions and occupations Paraguay expedition ( 1858 ) First Occupation of Cuba ( 1899 -- 1902 ) Panama -- Colombia separation and Hay -- Bunau - Varilla
Treaty ( 1903 ) Second Occupation of Cuba ( 1906 -- 09 ) Occupations of Honduras Occupation of Nicaragua (
</item 4>

<item 5>
Roosevelt Corollary
This was used to justify Operation PBSUCCESS that deposed the democratically elected President Jacobo Árbenz and installed the military regime of Carlos
Castillo Armas , the first in a series of military dictators in the country . Criticism ( edit ) The argument made by Mitchener and Weidenmier in 2006 in
support of the Roosevelt Corollary to the Monroe Doctrine has been criticized on the grounds that it `` represent ( s ) the one - sided approach that some
scholars bring to the study of imperialistic and hegemonic interventions and also highlight how arguments for the general utility of imperialism are
increasingly made and accepted . '' Christopher Coyne and Stephen Davies argue that a foreign policy modeled on the Roosevelt Corollary leads to negative
consequences both in national security terms and in terms of its effect on domestic politics . Critics , such as Noam Chomsky , have argued that the Roosevelt
Corollary was merely a more explicit imperialist threat , building on the Monroe Doctrine , indicating that the U.S. would intervene not only in defense of
South American states in the face of European imperialism , but would also use its muscle to obtain concessions and privileges for American corporations .
</item 5>

<item 6>
Roosevelt Corollary
Serge Ricard of the University of Paris goes even further , stating that the Roosevelt Corollary was not merely an addendum to the earlier Monroe Doctrine ,
through which the U.S. pledged to protect the Americas from European imperialist interventions . Rather , the Roosevelt Corollary was `` an entirely new
diplomatic tenet that epitomized his ' big stick ' approach to foreign policy . '' In other words , while the Monroe Doctrine sought to bar entry to the
European empires , the Roosevelt Corollary announced the United States intention to take their place . See also ( edit ) International relations portal
Government of the United States portal United States portal Wikisource has original text related to this article : Roosevelt Corollary History of the United
States ( 1865 -- 1918 ) United States territorial acquisitions New Imperialism Dollar Diplomacy Big Stick ideology Drago Doctrine Bush Doctrine References (
edit ) Jump up ^ Theodore Roosevelt , 26th President . Prod . David Grubin . By David Grubin and Geoffrey C. Ward . Perf . Walter LaFeber . David Grubin
Productions , Inc. , 1996 .
</item 6>

<item 7>
Roosevelt Corollary
Contents ( hide ) 1 Use 1.1 Shift to the `` Good Neighbor '' policy 2 Criticism 3 See also 4 References 5 Bibliography Use ( edit ) See also : Overseas
interventions of the United States The Roosevelt Corollary was an addition to the Monroe Doctrine ; however , it could be seen as a departure . While the Monroe
Doctrine said European countries should stay out of Latin America , the Roosevelt Corollary took this further to say he had the right to exercise military force
in Latin American countries to keep European countries out . Historian Walter LaFeber wrote ( Roosevelt ) essentially turns the Monroe Doctrine on its head and
toes says the Europeans should stay out , but the United States has the right , under the doctrine , to go in to exercise police power to keep the Europeans out
of the way .
</item 7>

<item 8>
Roosevelt Corollary
It is a very nice twist on the Monroe Doctrine , and of course , it becomes very , very important because over the next 15 to 20 years , the United States will
move into Latin America about a dozen times with military force , to the point where the United States Marines become known in the area as `` State Department
Troops '' because they are always moving in to protect State Department interests and State Department policy in the Caribbean . So what Roosevelt does here ,
by redefining the Monroe Doctrine , turns out to be very historical , and it leads the United States into a period of confrontation with peoples in the
Caribbean and Central America , that was an imperative part of American imperialism . Roosevelt first used the Corollary to act in the Dominican Republic in
1904 , which at the time was severely indebted and becoming a failed state . The United States dispatched two warships and demanded the customs house be turned
over to U.S. negotiators , who then used a percentage of the proceeds to pay foreign creditors . This model -- in which United States advisors worked to
stabilize Latin American nations through temporary protectorates , staving off European action -- became known as `` dollar diplomacy '' .
</item 8>

<item 9>
Roosevelt Corollary
In 1934 , Franklin D. Roosevelt further renounced interventionism and established his `` Good Neighbor policy '' that led to the annulment of the Platt
Amendment by the Treaty of Relations with Cuba in 1934 , and the negotiation of compensation for Mexico 's nationalization of foreign - owned oil assets in 1938
. Indeed , leaving unchallenged the emergence of dictatorships like that of Fulgencio Batista in Cuba , Rafael Leonidas Trujillo in the Dominican Republic ,
Anastasio Somoza in Nicaragua , and François Duvalier in Haiti were each considered to be `` Frankenstein dictators '' due to the mishandlings of the American
occupations in the countries . The era of the Good Neighbor Policy ended with the ramp - up of the Cold War in 1945 , as the United States felt there was a
greater need to protect the western hemisphere from Soviet influence . In 1954 , Secretary of State John Foster Dulles invoked the Monroe Doctrine and the
Roosevelt Corollary at the Tenth Pan-American Conference in Caracas , denouncing the intervention of Soviet Communism in Guatemala .
</item 9>

<item 10>
Roosevelt Corollary
The Dominican experiment , like most other `` dollar diplomacy '' arrangements , proved temporary and untenable , and the United States launched a larger
military intervention in 1912 . U.S. Presidents also cited the Roosevelt Corollary as justification for U.S. intervention in Cuba ( 1906 -- 1909 ) , Nicaragua (
1909 -- 1910 , 1912 -- 1925 and 1926 -- 1933 ) , Haiti ( 1915 -- 1934 ) , and the Dominican Republic ( 1916 -- 1924 ) . Shift to the `` Good Neighbor '' policy
( edit ) Main article : Good Neighbor policy In 1928 , under President Calvin Coolidge , the Clark Memorandum -- often seen as a partial repudiation of the
Roosevelt Corollary -- stated that the U.S. did not have the right to intervene when there was a threat by European powers . Herbert Hoover also helped move the
U.S. away from the imperialist tendencies of the Roosevelt Corollary by going on good - will tours , withdrawing troops from Nicaragua and Haiti , and
abstaining from intervening in the internal affairs of neighboring countries .
</item 10>
=== END ITEM id=55dab9f2700570fa ===

=== ITEM id=17786ff876b3fb74 ===
Gold sentences and facts:
Sentence 0: ( hide ) Climate data for The South Pole Month Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8 ( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( − 14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.0 ( − 63.4 ) − 53.6 ( − 64.5 ) − 54.5 ( − 66.1 ) − 55.2 ( − 67.4 ) − 54.9 ( − 66.8 ) − 54.4 ( − 65.9 ) − 48.4 ( − 55.1 ) − 36.2 ( − 33.2 ) − 26.3 ( − 15.3 ) − 45.8 ( − 50.4 ) Daily mean ° C ( ° F ) − 28.4 ( − 19.1 ) − 40.9 ( − 41.6 ) − 53.7 ( − 64.7 ) − 57.8 ( − 72 ) − 58.0 ( − 72.4 ) − 58.9 ( − 74 ) − 59.8 ( − 75.6 ) − 59.7 ( − 75.5 ) − 59.1 ( − 74.4 ) − 51.6 ( − 60.9 ) − 38.2 ( − 36.8 ) − 28.0 ( − 18.4 ) − 49.5 ( − 57.1 ) Average low ° C ( ° F ) − 29.6 ( − 21.3 ) − 43.1 ( − 45.6 ) − 56.8 ( − 70.2 ) − 60.9 ( − 77.6 ) − 61.5 ( − 78.7 ) − 62.8 ( − 81 ) − 63.4 ( − 82.1 ) − 63.2 ( − 81.8 ) − 61.7 ( − 79.1 ) − 54.3 ( − 65.7 ) − 40.1 ( − 40.2 ) − 29.1 ( − 20.4 ) − 52.2 ( − 62 ) Record low ° C ( ° F ) − 41.1 ( − 42 ) − 58.9 ( − 74 ) − 71.1 ( − 96 ) − 75.0 ( − 103 ) − 78.3 ( − 108.9 ) − 82.8 ( − 117 ) − 80.6 ( − 113.1 ) − 79.3 ( − 110.7 ) − 79.4 ( − 110.9 ) − 72.0 ( − 97.6 ) − 55.0 ( − 67 ) − 41.1 ( − 42 ) − 82.8 ( − 117 ) Average precipitation mm ( inches ) 0.3 ( 0.012 ) 0.6 ( 0.024 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) 0.2 ( 0.008 ) 0.1 ( 0.004 ) Trace Trace 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.1 ( 0.004 ) 0.3 ( 0.012 ) 2.3 ( 0.091 ) Average precipitation days ( ≥ 0.1 mm ) 0.2 0.3 0.2 0.0 0.2 0.1 0.0 0.0 0.1 0.1 0.1 0.3 1.6 Average snowy days 22.0 19.6 13.6 11.4 17.2 17.3 18.2 17.5 11.7 16.7 16.9 20.6 203.0 Mean monthly sunshine hours 406.1 497.2 195.3 0.0 0.0 0.0 0.0 0.0 34.1 390.6 558.0 616.9 2,698.2 Mean daily sunshine hours 13.1 17.6 6.3 0.0 0.0 0.0 0.0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst ( precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 )
  Fact 0.0: The climate data table is for The South Pole.
  Fact 0.1: The record high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -14.4 (6.1) in January, -20.6 (-5.1) in February, -26.7 (-16.1) in March, -27.8 (-18) in April, -25.1 (-13.2) in May, -28.8 (-19.8) in June, -33.9 (-29) in July, -32.8 (-27) in August, -29.3 (-20.7) in September, -25.1 (-13.2) in October, -18.9 (-2) in November and -12.3 (9.9) in December.
  Fact 0.2: The record high temperature for the year at The South Pole is -12.3 (9.9) degrees Celsius (Fahrenheit).
  Fact 0.3: The average high temperatures at The South Pole in degrees Celsius (Fahrenheit) are -26.0 (-14.8) in January, -37.9 (-36.2) in February, -49.6 (-57.3) in March, -53.0 (-63.4) in April, -53.6 (-64.5) in May, -54.5 (-66.1) in June, -55.2 (-67.4) in July, -54.9 (-66.8) in August, -54.4 (-65.9) in September, -48.4 (-55.1) in October, -36.2 (-33.2) in November and -26.3 (-15.3) in December.
  Fact 0.4: The average high temperature for the year at The South Pole is -45.8 (-50.4) degrees Celsius (Fahrenheit).
  Fact 0.5: The daily mean temperatures at The South Pole in degrees Celsius (Fahrenheit) are -28.4 (-19.1) in January, -40.9 (-41.6) in February, -53.7 (-64.7) in March, -57.8 (-72) in April, -58.0 (-72.4) in May, -58.9 (-74) in June, -59.8 (-75.6) in July, -59.7 (-75.5) in August, -59.1 (-74.4) in September, -51.6 (-60.9) in October, -38.2 (-36.8) in November and -28.0 (-18.4) in December.
  Fact 0.6: The daily mean temperature for the year at The South Pole is -49.5 (-57.1) degrees Celsius (Fahrenheit).
  Fact 0.7: The average low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -29.6 (-21.3) in January, -43.1 (-45.6) in February, -56.8 (-70.2) in March, -60.9 (-77.6) in April, -61.5 (-78.7) in May, -62.8 (-81) in June, -63.4 (-82.1) in July, -63.2 (-81.8) in August, -61.7 (-79.1) in September, -54.3 (-65.7) in October, -40.1 (-40.2) in November and -29.1 (-20.4) in December.
  Fact 0.8: The average low temperature for the year at The South Pole is -52.2 (-62) degrees Celsius (Fahrenheit).
  Fact 0.9: The record low temperatures at The South Pole in degrees Celsius (Fahrenheit) are -41.1 (-42) in January, -58.9 (-74) in February, -71.1 (-96) in March, -75.0 (-103) in April, -78.3 (-108.9) in May, -82.8 (-117) in June, -80.6 (-113.1) in July, -79.3 (-110.7) in August, -79.4 (-110.9) in September, -72.0 (-97.6) in October, -55.0 (-67) in November and -41.1 (-42) in December.
  Fact 0.10: The record low temperature for the year at The South Pole is -82.8 (-117) degrees Celsius (Fahrenheit).
  Fact 0.11: The average precipitation at The South Pole in millimeters (inches) is 0.3 (0.012) in January, 0.6 (0.024) in February, 0.2 (0.008) in March, 0.1 (0.004) in April, 0.2 (0.008) in May, 0.1 (0.004) in June, Trace in July, Trace in August, 0.1 (0.004) in September, 0.1 (0.004) in October, 0.1 (0.004) in November and 0.3 (0.012) in December.
  Fact 0.12: The average precipitation for the year at The South Pole is 2.3 (0.091) millimeters (inches).
  Fact 0.13: The average number of days with precipitation of at least 0.1 mm at The South Pole is 0.2 in January, 0.3 in February, 0.2 in March, 0.0 in April, 0.2 in May, 0.1 in June, 0.0 in July, 0.0 in August, 0.1 in September, 0.1 in October, 0.1 in November and 0.3 in December.
  Fact 0.14: The average number of days with precipitation of at least 0.1 mm at The South Pole is 1.6 for the year.
  Fact 0.15: The average number of snowy days at The South Pole is 22.0 in January, 19.6 in February, 13.6 in March, 11.4 in April, 17.2 in May, 17.3 in June, 18.2 in July, 17.5 in August, 11.7 in September, 16.7 in October, 16.9 in November and 20.6 in December.
  Fact 0.16: The average number of snowy days at The South Pole is 203.0 for the year.
  Fact 0.17: The mean monthly sunshine hours at The South Pole are 406.1 in January, 497.2 in February, 195.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 34.1 in September, 390.6 in October, 558.0 in November and 616.9 in December.
  Fact 0.18: The mean monthly sunshine hours at The South Pole total 2,698.2 for the year.
  Fact 0.19: The mean daily sunshine hours at The South Pole are 13.1 in January, 17.6 in February, 6.3 in March, 0.0 in April, 0.0 in May, 0.0 in June, 0.0 in July, 0.0 in August, 1.1 in September, 12.6 in October, 18.6 in November and 19.9 in December.
  Fact 0.20: The mean daily sunshine hours at The South Pole are 7.4 for the year.
  Fact 0.21: Source 1 for the data is Pogoda.ru.net, for temperatures from 1981 to 2010 and extremes from 1957 to present.
  Fact 0.22: Source 2 for the data is Deutscher Wetterdienst, for precipitation from 1957 to 1988 and sun from 1978 to 1993.
  Fact 0.23: Source 2 also includes NOAA, for snowy days data from 1961 to 1988.

Retrieved items:
<item 1>
South Pole
Centrifugal force from the spin of the planet pulls the atmosphere toward the equator . The South Pole is colder than the North Pole because of the elevation
difference . The North Pole is a few feet from sea level . In midsummer , as the sun reaches its maximum elevation of about 23.5 degrees , high temperatures at
the South Pole in January average at − 25.9 ° C ( − 15 ° F ) . As the six - month `` day '' wears on and the sun gets lower , temperatures drop as well : they
reach − 45 ° C ( − 49 ° F ) around sunset ( late March ) and sunrise ( late September ) . In midwinter , the average temperature remains steady at around − 58 °
C ( − 72 ° F ) .
</item 1>

<item 2>
South Pole
The polar ice sheet is moving at a rate of roughly 10 metres per year in a direction between 37 ° and 40 ° west of grid north , down towards the Weddell Sea .
Therefore , the position of the station and other artificial features relative to the geographic pole gradually shift over time . The Geographic South Pole is
marked by a stake in the ice alongside a small sign ; these are repositioned each year in a ceremony on New Year 's Day to compensate for the movement of the
ice . The sign records the respective dates that Roald Amundsen and Robert F. Scott reached the Pole , followed by a short quotation from each man , and gives
the elevation as `` 9,301 FT . '' . A new marker stake is designed and fabricated each year by staff at the site . Ceremonial South Pole ( edit ) The Ceremonial
South Pole is an area set aside for photo opportunities at the South Pole Station . It is located some meters from the Geographic South Pole , and consists of a
metallic sphere on a short bamboo pole , surrounded by the flags of the original Antarctic Treaty signatory states .
</item 2>

<item 3>
South Pole
5 ) − 59.1 ( − 74.4 ) − 51.6 ( − 60.9 ) − 38.2 ( − 36.8 ) − 28.0 ( − 18.4 ) − 49.5 ( − 57.1 ) Average low ° C ( ° F ) − 29.6 ( − 21.3 ) − 43.1 ( − 45.6 ) − 56.8
( − 70.2 ) − 60.9 ( − 77.6 ) − 61.5 ( − 78.7 ) − 62.8 ( − 81 ) − 63.4 ( − 82.1 ) − 63.2 ( − 81.8 ) − 61.7 ( − 79.1 ) − 54.3 ( − 65.7 ) − 40.1 ( − 40.2 ) − 29.1
( − 20.4 ) − 52.2 ( − 62 ) Record low ° C ( ° F ) − 41.1 ( − 42 ) − 58.
</item 3>

<item 4>
South Pole
More recent buildings are raised on stilts so that the snow does not build up against the sides of them . ( hide ) Climate data for The South Pole Month Jan Feb
Mar Apr May Jun Jul Aug Sep Oct Nov Dec Year Record high ° C ( ° F ) − 14.4 ( 6.1 ) − 20.6 ( − 5.1 ) − 26.7 ( − 16.1 ) − 27.8 ( − 18 ) − 25.1 ( − 13.2 ) − 28.8
( − 19.8 ) − 33.9 ( − 29 ) − 32.8 ( − 27 ) − 29.3 ( − 20.7 ) − 25.1 ( − 13.2 ) − 18.9 ( − 2 ) − 12.3 ( 9.9 ) − 12.3 ( 9.9 ) Average high ° C ( ° F ) − 26.0 ( −
14.8 ) − 37.9 ( − 36.2 ) − 49.6 ( − 57.3 ) − 53.
</item 4>

<item 5>
South Pole
The highest temperature ever recorded at the Amundsen -- Scott South Pole Station was − 12.3 ° C ( 9.9 ° F ) on Christmas Day , 2011 , and the lowest was − 82.8
° C ( − 117.0 ° F ) on June 23 , 1982 ( for comparison , the lowest temperature directly recorded anywhere on earth was − 89.2 ° C ( − 128.6 ° F ) at Vostok
Station on July 21 , 1983 , though − 93.2 ° C ( − 135.8 ° F ) was measured indirectly by satellite in East Antarctica between Dome A and Dome F in August 2010 )
. The South Pole has a desert climate , receiving very little precipitation . Air humidity is near zero . However , high winds can cause the blowing of snowfall
, and the accumulation of snow amounts to about 20 cm ( 8 in ) per year . The former dome seen in pictures of the Amundsen -- Scott station is partially buried
due to snow storms , and the entrance to the dome had to be regularly bulldozed to uncover it .
</item 5>

<item 6>
South Pole
However , it is always low in the sky , reaching a maximum of 23.5 ° in December . Much of the sunlight that does reach the surface is reflected by the white
snow . This lack of warmth from the sun , combined with the high altitude ( about 2,800 metres ( 9,200 ft ) ) , means that the South Pole has one of the coldest
climates on Earth ( though it is not quite the coldest ; that record goes to the region in the vicinity of the Vostok Station , also in Antarctica , which lies
at a higher elevation ) . Temperatures at the South Pole are much lower than at the North Pole , primarily because the South Pole is located at altitude in the
middle of a continental land mass , while the North Pole is at sea level in the middle of an ocean , which acts as a reservoir of heat . The South Pole is at an
altitude of 9,300 feet ( 2,800 m ) but feels like 11,000 feet ( 3,400 m ). . Centrifugal force from the spin of the planet pulls the atmosphere toward the
equator .
</item 6>

<item 7>
South Pole
The geographic coordinates of the South Pole are usually given simply as 90 ° S , since its longitude is geometrically undefined and irrelevant . When a
longitude is desired , it may be given as 0 ° . At the South Pole , all directions face north . For this reason , directions at the Pole are given relative to
`` grid north '' , which points northwards along the prime meridian . Along tight latitude circles , clockwise is east , and counterclockwise is west , opposite
to the North Pole . The Geographic South Pole is located on the continent of Antarctica ( although this has not been the case for all of Earth 's history
because of continental drift ) . It sits atop a featureless , barren , windswept and icy plateau at an altitude of 2,835 metres ( 9,301 ft ) above sea level ,
and is located about 1,300 km ( 800 mi ) from the nearest open sea at Bay of Whales . The ice is estimated to be about 2,700 metres ( 9,000 ft ) thick at the
Pole , so the land surface under the ice sheet is actually near sea level .
</item 7>

<item 8>
North Pole
7 ( − 3.7 ) Record low ° C ( ° F ) − 47 ( − 53 ) − 50 ( − 58 ) − 50 ( − 58 ) − 41 ( − 42 ) − 24 ( − 11 ) − 12 ( 10 ) − 2 ( 28 ) − 12 ( 10 ) − 31 ( − 24 ) − 41 (
− 42 ) − 41 ( − 42 ) − 47 ( − 53 ) − 50 ( − 58 ) Average relative humidity ( % ) 83.5 83.0 83.0 85.0 87.5 90.0 90.0 89.5 88.0 84.5 83.0 83.0 85.83 Source :
Weatherbase Flora and fauna ( edit ) Polar bears are believed rarely to travel beyond about 82 ° North owing to the scarcity of food , though tracks have been
seen in the vicinity of the North Pole , and a 2006 expedition reported sighting a polar bear just 1 mi ( 1.6 km ) from the Pole .
</item 8>

<item 9>
South Pole
This is because the US flies its resupply missions ( `` Operation Deep Freeze '' ) out of McMurdo Station , which is supplied from Christchurch , New Zealand .
Flora and fauna ( edit ) Due to its exceptionally harsh climate , there are no native resident plants or animals at the South Pole . Remarkably , though , off -
course south polar skuas and snow petrels are occasionally seen there . In 2000 it was reported that microbes had been detected living in the South Pole ice .
See also ( edit ) Antarctica portal List of Antarctic expeditions North Pole South Pole Telescope References ( edit ) Jump up ^ `` Moving the South Pole ''
Archived 2011 - 07 - 16 at the Wayback Machine. , NASA Quest ^ Jump up to : Amundsen -- Scott South Pole Station , National Science Foundation , Office of Polar
Programs Jump up ^ `` Where is the real Pole really ? '' . Retrieved 2008 - 03 - 25 . ^ Jump up to : `` Marker makes annual move '' , page 6 , Antarctic Sun .
January 8 , 2006 ; McMurdo Station , Antarctica .
</item 9>

<item 10>
South Pole
0 0.0 1.1 12.6 18.6 19.9 7.4 Source # 1 : Pogoda.ru.net ( temperatures , 1981 -- 2010 , extremes 1957 -- present ) Source # 2 : Deutscher Wetterdienst (
precipitation 1957 -- 1988 and sun 1978 -- 1993 ) , NOAA ( snowy days data , 1961 -- 1988 ) Time ( edit ) In most places on Earth , local time is determined by
longitude , such that the time of day is more - or-less synchronised to the position of the sun in the sky ( for example , at midday the sun is roughly at its
highest ) . This line of reasoning fails at the South Pole , where the sun rises and sets only once per year , and all lines of longitude , and hence all time
zones , converge . There is no a priori reason for placing the South Pole in any particular time zone , but as a matter of practical convenience the Amundsen --
Scott South Pole Station keeps New Zealand Time ( UTC + 12 ) .
</item 10>
=== END ITEM id=17786ff876b3fb74 ===

=== ITEM id=b5a66b031d5c9c5f ===
Gold sentences and facts:
Sentence 0: Rigor mortis is very important in meat technology . The onset of rigor mortis and its resolution partially determine the tenderness of meat . If the post-slaughter meat is immediately chilled to 15 ° C ( 59 ° F ) , a phenomenon known as cold shortening occurs , whereby the muscle sarcomeres shrink to a third of their original length .
  Fact 0.0: Rigor mortis is very important in meat technology
  Fact 0.1: The onset of rigor mortis partially determines the tenderness of meat
  Fact 0.2: The resolution of rigor mortis partially determines the tenderness of meat
  Fact 0.3: If post-slaughter meat is immediately chilled to 15 °C (59 °F), a phenomenon known as cold shortening occurs
  Fact 0.4: Cold shortening is the phenomenon that occurs when post-slaughter meat is immediately chilled to 15 °C (59 °F)
  Fact 0.5: In cold shortening, the muscle sarcomeres shrink to a third of their original length

Retrieved items:
<item 1>
Graph query row:
entity: Rigor mortis
relationship: RELATED_TO
related: meat technology
</item 1>

<item 2>
Graph query row:
entity: cold shortening
relationship: RELATED_TO
related: electrical stimulation
</item 2>

<item 3>
Rigor mortis
The calcium ions trigger powerful muscle contraction aided by ATP molecules . To prevent cold shortening , a process known as electrical stimulation is carried
out , especially in beef carcasses , immediately after slaughter and skinning . In this process , the carcass is stimulated with alternating current , causing
it to contract and relax , which depletes the ATP reserve from the carcass and prevents cold shortening . Application in forensic pathology ( edit ) The degree
of rigor mortis may be used in forensic pathology , to determine the approximate time of death . A dead body holds its position as rigor mortis sets in . If the
body is moved after death , but before rigor mortis begins , forensic techniques such as livor mortis can be applied . If the position in which a body is found
does not match the location where it is found ( for example , if it is flat on its back with one arm sticking straight up ) , that could mean someone moved it .
Several factors also affect the progression of rigor mortis , and investigators take these into account when estimating the time of death . One such factor is
the ambient temperature .
</item 3>

<item 4>
Rigor mortis
The sequence may be due to different lactic acid levels among different muscles , which is directly related to the difference in glycogen levels and different
types of muscle fibers . Rigor mortis then spreads to the other muscles , including the internal organs , within the next four to six hours . The onset of rigor
mortis is affected by the individual 's age , sex , physical condition , and muscular build . Rigor mortis may not be perceivable in many infant and child
corpses due to their smaller muscle mass . Applications in meat industry ( edit ) Rigor mortis is very important in meat technology . The onset of rigor mortis
and its resolution partially determine the tenderness of meat . If the post-slaughter meat is immediately chilled to 15 ° C ( 59 ° F ) , a phenomenon known as
cold shortening occurs , whereby the muscle sarcomeres shrink to a third of their original length . Cold shortening is caused by the release of stored calcium
ions from the sarcoplasmic reticulum of muscle fibers , in response to the cold stimulus . The calcium ions trigger powerful muscle contraction aided by ATP
molecules .
</item 4>

<item 5>
Graph query row:
entity: Rigor mortis
relationship: RELATED_TO
related: cold shortening
</item 5>

<item 6>
Graph query row:
entity: Rigor mortis
relationship: RELATED_TO
related: tenderness of meat
</item 6>

<item 7>
Rigor mortis
One such factor is the ambient temperature . When conditions are warm , the onset and pace of rigor mortis are sped up by providing a conducive environment for
the metabolic processes that cause decay . Low temperatures , however , slow them down . Therefore , for a person who dies outside in frozen conditions rigor
mortis may last several days more than normal , so investigators may have to abandon it as a tool for determining time of death . See also ( edit ) Cadaveric
spasm References ( edit ) ^ Jump up to : Saladin , K.S. 2010 . Anatomy & Physiology : 6th edition . McGraw - Hill . ^ Jump up to : Hall , John E. , and Arthur
C. Guyton . Guyton and Hall Textbook of Medical Physiology . Philadelphia , PA : Saunders / Elsevier , 2011 . MD Consult . Web . 26 Jan. 2015 . Jump up ^
Fremery , Donald ( 3 February 1959 ) . `` Biochemistry of Chicken Muscle as Related to Rigor Mortis and Tenderization '' . Journal of Food Sciences . 25 ( 1 ) :
73 -- 87 .
</item 7>

<item 8>
Rigor mortis
Rigor mortis - wikipedia Rigor mortis Jump to : navigation , search This article is about the sign of death . For other uses , see Rigor mortis ( disambiguation
) . Stages of death Pallor mortis Algor mortis Rigor mortis Livor mortis Putrefaction Decomposition Skeletonization Fossilization Rigor mortis ( Latin : rigor
`` stiffness '' , mortis `` of death '' ) or postmortem rigidity , the third stage of death , is one of the recognizable signs of death , caused by chemical
changes in the muscles post mortem , which cause the limbs of the corpse to stiffen . In humans , rigor mortis can occur as soon as 4 hours post mortem .
Contents ( hide ) 1 Physiology 2 Physical changes 3 Applications in meat industry 4 Application in forensic pathology 5 See also 6 References 7 Bibliography
Physiology ( edit ) After death , respiration in an organism ceases , depleting the source of oxygen used in the making of adenosine triphosphate ( ATP ) .
</item 8>

<item 9>
Rigor mortis
In rigor mortis myosin heads continue binding with the active sites of actin proteins via adenosine diphosphate ( ADP ) , and the muscle is unable to relax
until further enzyme activity degrades the complex . Normal relaxation would occur by replacing ADP with ATP , which would destabilize the myosin - actin bond
and break the cross-bridge . However , as ATP is absent , there must be a breakdown of muscle tissue by enzymes ( endogenous or bacterial ) during decomposition
. As part of the process of decomposition , the myosin heads are degraded by the enzymes , allowing the muscle contraction to release and the body to relax .
Decomposition of the myofilaments occurs 48 to 60 hours after the peak of rigor mortis , which occurs approximately 13 hours after death . Physical Changes (
edit ) At the time of death , a condition called `` primary flaccidity '' occurs . Following this , the muscles stiffen in rigor mortis . All muscles in the
body are affected . Starting between two and six hours following death , rigor mortis begins with the eyelids , neck , and jaw .
</item 9>

<item 10>
Rigor mortis
Journal of Food Sciences . 25 ( 1 ) : 73 -- 87 . doi : 10.1111 / j. 1365 - 2621.1960. tb17938. x . Retrieved 2 December 2015 . Jump up ^ About.com Jump up ^ ``
Classroom Resources - Argonne National Laboratory '' . Jump up ^ `` Rigor Mortis and Other Postmortem Changes - Burial , Body , Life , Cause , Time , Person ,
Human , Putrefaction . '' Encyclopedia of Death and Dying. 2011 . Web . 4 December 2011 . <
http://www.deathreference.com/Py-Se/Rigor-Mortis-and-Other-Postmortem-Changes.html > . Jump up ^ `` New Zealand Journal of Agricultural Research '' . Jump up ^
Peress , Robin . `` Discovery Health `` Rigor Mortis at the Crime Scene '' '' Discovery Health `` Health Guides '' Discovery Fit & Health , 2011 . Web . 4
December 2011 .
</item 10>
=== END ITEM id=b5a66b031d5c9c5f ===

=== ITEM id=fc423a524ddac3f2 ===
Gold sentences and facts:
Sentence 0: A regulatory site ( also known as an `` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes conformational changes which changes the enzyme 's activity .
  Fact 0.0: A regulatory site is also known as an allosteric site.
  Fact 0.1: A regulatory site is a site on an allosteric protein.
  Fact 0.2: A modulator molecule binds to a regulatory site on an allosteric protein.
  Fact 0.3: A modulator molecule is also known as an allosteric regulator.
  Fact 0.4: The ligand-binding site on a receptor or enzyme is distinct from the active site.
  Fact 0.5: Allosteric modulators alter enzyme activity by binding to the regulatory site.
  Fact 0.6: Some allosteric regulators activate enzymes.
  Fact 0.7: Other allosteric regulators inhibit enzymes.
  Fact 0.8: The working hypothesis is that when allosteric regulators bind to the regulatory site, the protein undergoes conformational changes.
  Fact 0.9: The working hypothesis is that these conformational changes change the enzyme's activity.

Retrieved items:
<item 1>
ATP synthase
Because of its rotating subunit ATP Synthase is a molecular machine . Contents ( hide ) 1 Nomenclature 2 Structure and function 2.1 F region 2.2 F region 3
Binding model 4 Physiological role 5 Evolution 6 In different species 6.1 E. coli 6.2 Yeast 6.3 Plant 6.4 Bovine 6.5 Human 7 See also 8 References 9 Further
reading 10 External links Nomenclature ( edit ) The F fraction derives its name from the term `` Fraction 1 '' and F ( written as a subscript letter `` o '' ,
not `` zero '' ) derives its name from being the binding fraction for oligomycin , a type of naturally - derived antibiotic that is able to inhibit the F unit
of ATP synthase . These functional regions consist of different protein subunits -- refer to tables . Structure and function ( edit ) Bovine mitochondrial ATP
synthase . The F , F , axle , and stator regions are color coded magenta , green , orange , and cyan respectively . Simplified model of F F - ATPase alias ATP
synthase of E. coli .
</item 1>

<item 2>
ATP synthase
Jump up ^ Ahmad Z , Okafor F , Laughlin TF . `` Role of Charged Residues in the Catalytic Sites of Escherichia coli ATP Synthase '' . Journal of Amino Acids.
2011 : 785741 . doi : 10.4061 / 2011 / 785741 . PMID 22312470 . Jump up ^ Devenish RJ , Prescott M , Roucou X , Nagley P ( May 2000 ) . `` Insights into ATP
synthase assembly and function through the molecular genetic manipulation of subunits of the yeast mitochondrial enzyme complex '' . Biochimica et Biophysica
Acta. 1458 ( 2 -- 3 ) : 428 -- 42 . doi : 10.1016 / S0005 - 2728 ( 00 ) 00092 - X . PMID 10838056 . Jump up ^ Kabaleeswaran V , Puri N , Walker JE , Leslie AG ,
Mueller DM ( November 2006 ) . `` Novel features of the rotary catalytic mechanism revealed in the structure of yeast F ATPase '' .
</item 2>

<item 3>
ATP synthase
The cryo - EM model of ATP synthase suggests that the peripheral stalk is a flexible structure that wraps around the complex as it joins F to F. Under the right
conditions , the enzyme reaction can also be carried out in reverse , with ATP hydrolysis driving proton pumping across the membrane . The binding change
mechanism involves the active site of a β subunit 's cycling between three states . In the `` loose '' state , ADP and phosphate enter the active site ; in the
adjacent diagram , this is shown in pink . The enzyme then undergoes a change in shape and forces these molecules together , with the active site in the
resulting `` tight '' state ( shown in red ) binding the newly produced ATP molecule with very high affinity . Finally , the active site cycles back to the open
state ( orange ) , releasing ATP and binding more ADP and phosphate , ready for the next cycle of ATP production . Physiological role ( edit ) Like other
enzymes , the activity of F F ATP synthase is reversible .
</item 3>

<item 4>
ATP synthase
`` Atomic model for the dimeric FO region of mitochondrial ATP synthase '' . Science . 358 ( 6365 ) : 936 -- 40 . doi : 10.1126 / science. aao4815 . PMID
29074581 . Jump up ^ Gresser MJ , Myers JA , Boyer PD ( October 1982 ) . `` Catalytic site cooperativity of beef heart mitochondrial F adenosine triphosphatase
. Correlations of initial velocity , bound intermediate , and oxygen exchange measurements with an alternating three - site model '' . The Journal of Biological
Chemistry . 257 ( 20 ) : 12030 -- 8 . PMID 6214554 . Jump up ^ Nakamoto RK , Baylis Scanlon JA , Al - Shawi MK ( August 2008 ) . `` The rotary mechanism of the
ATP synthase '' . Archives of Biochemistry and Biophysics. 476 ( 1 ) : 43 -- 50 . doi : 10.1016 / j. abb. 2008.05. 004 . PMC 2581510 .
</item 4>

<item 5>
Regulatory site
Regulatory site - Wikipedia Regulatory site Jump to : navigation , search It has been suggested that this article be merged with Allosteric regulation . (
Discuss ) Proposed since October 2017 . This article does not cite any sources . Please help improve this article by adding citations to reliable sources .
Unsourced material may be challenged and removed . ( December 2009 ) ( Learn how and when to remove this template message ) A regulatory site ( also known as an
`` allosteric site '' ) is a site on an allosteric protein to which a modulator molecule ( or allosteric regulator ) binds . This ligand - binding site on a
receptor or enzyme is distinct from the active site . Allosteric modulators alter enzyme activity by binding to the regulatory site . Some allosteric regulators
activate enzymes , while others inhibit them . The working hypothesis is that when allosteric regulators bind to the regulatory site , the protein undergoes
conformational changes which changes the enzyme 's activity . This biochemistry article is a stub . You can help Wikipedia by expanding it .
</item 5>

<item 6>
ATP synthase
5.5 - 6 : Polymerization motors Dynamin Tubulin Enzymes Activity Active site Binding site Catalytic triad Oxyanion hole Enzyme promiscuity Catalytically perfect
enzyme Coenzyme Cofactor Enzyme catalysis Regulation Allosteric regulation Cooperativity Enzyme inhibitor Classification EC number Enzyme superfamily Enzyme
family List of enzymes Kinetics Enzyme kinetics Eadie -- Hofstee diagram Hanes -- Woolf plot Lineweaver -- Burk plot Michaelis -- Menten kinetics Types EC1
Oxidoreductases ( list ) EC2 Transferases ( list ) EC3 Hydrolases ( list ) EC4 Lyases ( list ) EC5 Isomerases ( list ) EC6 Ligases ( list ) Molecular and
Cellular Biology portal Retrieved from `` https://en.wikipedia.org/w/index.php?title=ATP_synthase&oldid=821081799 '' Categories : Enzymes Cellular respiration
Photosynthesis EC 3.6.
</item 6>

<item 7>
Regulatory site
This biochemistry article is a stub . You can help Wikipedia by expanding it . Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Regulatory_site&oldid=807494796 '' Categories : Protein structure Biochemistry stubs Hidden categories : Articles to
be merged from October 2017 All articles to be merged Articles lacking sources from December 2009 All articles lacking sources All stub articles Talk About
Wikipedia Add links This page was last edited on 28 October 2017 , at 09 : 38 . About Wikipedia
</item 7>

<item 8>
ATP synthase
ATP synthase - wikipedia ATP synthase Jump to : navigation , search ATP Synthase Molecular model of ATP synthase determined by X-ray crystallography . Stator is
not shown here . Identifiers EC number 3.6. 3.14 CAS number 9000 - 83 - 3 Databases IntEnz IntEnz view BRENDA BRENDA entry ExPASy NiceZyme view KEGG KEGG entry
MetaCyc metabolic pathway PRIAM profile PDB structures RCSB PDB PDBe PDBsum Gene Ontology AmiGO / QuickGO ( show ) Search PMC articles PubMed articles NCBI
proteins ATP synthase is an enzyme that creates the energy storage molecule adenosine triphosphate ( ATP ) . ATP is the most commonly used `` energy currency ''
of cells for most organisms . It is formed from adenosine diphosphate ( ADP ) and inorganic phosphate ( P ) .
</item 8>

<item 9>
ATP synthase
`` Partial resolution of the enzymes catalyzing oxidative phosphorylation. 8 . Properties of a factor conferring oligomycin sensitivity on mitochondrial
adenosine triphosphatase '' . The Journal of Biological Chemistry . 241 ( 10 ) : 2461 -- 6 . PMID 4223640 . Jump up ^ Mccarty RE ( November 1992 ) . `` A plant
biochemist 's view of H + - ATPases and ATP synthases `` . The Journal of Experimental Biology . 172 ( Pt 1 ) : 431 -- 441 . PMID 9874753 . Jump up ^ PDB : 5ARA
​ ; Zhou A , Rohou A , Schep DG , Bason JV , Montgomery MG , Walker JE , Grigorieff N , Rubinstein JL ( October 2015 ) . `` Structure and conformational states
of the bovine mitochondrial ATP synthase by cryo - EM '' . eLife. 4 : e10180 . doi : 10.7554 / eLife. 10180 . PMC 4718723 . PMID 26439008 .
</item 9>

<item 10>
ATP synthase
Proton and Sodium translocating F - type , V - type and A-type ATPases in OPM database The Nobel Prize in Chemistry 1997 to Paul D. Boyer and John E. Walker for
the enzymatic mechanism of synthesis of ATP ; and to Jens C. Skou , for discovery of an ion - transporting enzyme , Na + , K + - ATPase . Harvard Multimedia
Production Site -- Videos -- ATP synthesis animation David Goodsell : `` ATP Synthase - Molecule of the Month '' Membrane transport protein : ion pumps ,
ATPases / ATP synthase ( TC 3A2 - 3A3 ) F - , V - , and A-type ATPase ( 3. A.
</item 10>
=== END ITEM id=fc423a524ddac3f2 ===
