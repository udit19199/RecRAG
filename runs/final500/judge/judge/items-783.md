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

This file has 7 items.

=== ITEM id=ccca502a5e1047a0 ===
Gold sentences and facts:
Sentence 0: In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or qualitative considerations , rather than easily measurable and quantifiable factors .
  Fact 0.0: In negotiation theory, the best alternative to a negotiated agreement (BATNA) is the most advantageous course of action a party can take if negotiations fail and no agreement can be reached.
  Fact 0.1: BATNA is the key focus of a successful negotiator.
  Fact 0.2: BATNA is the driving force behind a successful negotiator.
  Fact 0.3: A party should generally not accept a resolution worse than its BATNA.
  Fact 0.4: Care should be taken to ensure that deals are accurately valued.
  Fact 0.5: Deal valuation should take into account all considerations.
  Fact 0.6: Relationship value is a consideration in valuing a deal.
  Fact 0.7: Time value of money is a consideration in valuing a deal.
  Fact 0.8: The likelihood that the other party will live up to its side of the bargain is a consideration in valuing a deal.
  Fact 0.9: The other considerations are often difficult to value.
  Fact 0.10: The other considerations are frequently based on uncertain or qualitative considerations.
  Fact 0.11: The other considerations are qualitative rather than easily measurable and quantifiable factors.

Retrieved items:
<item 1>
Best alternative to a negotiated agreement
History ( edit ) BATNA was developed by negotiation researchers Roger Fisher and William Ury of the Harvard Program on Negotiation ( PON ) , in their series of
books on Principled negotiation that started with Getting to YES , unwittingly duplicating the game theory concept of a disagreement point from bargaining
problems pioneered by Nobel Laureate John Forbes Nash decades earlier . A Nash Equilibrium is reached among a group of players when no player can benefit from
changing strategies if every other player sticks to their current strategy . For example , Amy and Phil are in Nash Equilibrium if Amy is making the best
decision she can , taking into account Phil 's decision , and Phil is making the best decision he can , taking into account Amy 's decision . Likewise , a group
of players are in Nash Equilibrium if each one is making the best decision that he or she can , taking into account the decisions of the others .
</item 1>

<item 2>
BATNA (best alternative to a negotiated agreement) is the most advantageous actionable option available if negotiations fail. Negotiators should generally
reject deals worse than their BATNA, while accurately weighing uncertain factors such as relationships, timing, and whether commitments will be honored. A
strong BATNA provides leverage, but people often overestimate their alternatives or fail to research them. Fisher and Ury recommend listing possible actions,
developing promising ideas into concrete options, and choosing the best. Negotiators should also consider the other party
</item 2>

<item 3>
Best alternative to a negotiated agreement
Retrieved 1 October 2012 . External links ( edit ) BATNA Explained by David Venter Alternative definition of BATNA Retrieved from ``
https://en.wikipedia.org/w/index.php?title=Best_alternative_to_a_negotiated_agreement&oldid=819966880 '' Categories : Negotiation Dispute resolution Business
terms Hidden categories : Articles needing more detailed references Articles needing additional references from July 2013 All articles needing additional
references All articles lacking reliable references Articles lacking reliable references from July 2013 All articles with unsourced statements Articles with
unsourced statements from October 2007 Articles lacking reliable references from February 2015 Wikipedia articles needing page number citations from July 2013
Talk Contents About Wikipedia Deutsch Español Հայերեն Polski Edit links This page was last edited on 12 January 2018 , at 08 : 19 . Text is available under the
Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the Terms of Use and Privacy Policy .
Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization .
</item 3>

<item 4>
Best alternative to a negotiated agreement
Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding . According to Fisher and Ury , when hard negotiators meet
soft negotiators , the hard negotiators usually win their position , but at the cost of potentially damaging the long term relationship between the parties .
Attractive alternatives are needed to develop a strong BATNA . In the best - selling book Getting to YES : Negotiating Agreement Without Giving In , the authors
give 3 suggestions of how to accomplish this : Inventing a list of actions one might take if no agreement is reached Converting some of the more promising ideas
and transforming them into tangible and partial alternatives Selecting the alternative that sounds best In negotiations involving different cultures , all
parties need to account for cultural cognitive behaviors and should not let judgments and biases affect the negotiation . The individual should be separate from
the objective . The purpose here , as Gulliver mentions , is for negotiation parties to be aware . Preparation at all levels , including prejudice - free
thoughts , emotion - free behavior , bias - free behavior are helpful according to Morris and Gelfand .
</item 4>

<item 5>
BATNA means the best actionable alternative if negotiation fails; negotiators should research options, assess uncertainty, and reject deals worse than it. U.S.
bar exams are jurisdiction-controlled, generally combining essays with NCBE tests: the 200-question MBE, MEE essays, MPT practical tasks, and usually the MPRE
ethics exam. The UBE combines MBE, MEE, and MPT for score portability; adoption remained incomplete in 2017. Admission rules and exam formats vary
</item 5>

<item 6>
Best alternative to a negotiated agreement
Best alternative to a negotiated agreement - wikipedia Best alternative to a negotiated agreement Jump to : navigation , search This article needs additional
citations for verification . Please help improve this article by adding citations to reliable sources . Unsourced material may be challenged and removed . (
July 2013 ) ( Learn how and when to remove this template message ) In negotiation theory , the best alternative to a negotiated agreement or BATNA is the most
advantageous alternative course of action a party can take if negotiations fail and an agreement can not be reached . BATNA is the key focus and the driving
force behind a successful negotiator . A party should generally not accept a worse resolution than its BATNA . Care should be taken , however , to ensure that
deals are accurately valued , taking into account all considerations , such as relationship value , time value of money and the likelihood that the other party
will live up to their side of the bargain . These other considerations are often difficult to value , since they are frequently based on uncertain or
qualitative considerations , rather than easily measurable and quantifiable factors .
</item 6>

<item 7>
The text contains Wikipedia page metadata for two unrelated topics: “Best alternative to a negotiated agreement” (BATNA), categorized under negotiation, dispute
resolution, and business terms, and “Battles of Saratoga,” categorized as a good article. It lists references and reliability warnings for the BATNA page, along
with edit dates, language links, licensing, and site terms.
</item 7>

<item 8>
Best alternative to a negotiated agreement
See also ( edit ) Conflict resolution research Opportunity cost References ( edit ) Jump up ^ Best Alternative to a Negotiated Agreement Negotiation Experts
Jump up ^ Best Alternative to a Negotiated Agreement YArooms Jump up ^ Negotiation , Readings , Exercises and Cases , Roy J. Lewicki ^ Jump up to : Nolan -
Haley , Jaqueline M ( 2001 ) . Alternative Dispute Resolution in a Nutshell . Thomson West . pp. 39 -- 50 . ISBN 978 - 0 - 314 - 18014 - 8 . ^ Jump up to :
Fisher and Ury , Roger and William ( 2011 ) . Getting to YES . Penguin Books . pp. 1 -- 170 . ISBN 978 - 0 - 14 - 311875 - 6 . Jump up ^ Gulliver , P.H ( 1979 )
. Disputes and Negotiation : A Cross Culture Perspective . Academic Press . p. 287 . Jump up ^ Morris , Michael W. ; Gelfand , Michele J. ( 2004 ) .
</item 8>

<item 9>
Best alternative to a negotiated agreement
Contents ( hide ) 1 Overview 2 History 3 See also 4 References 5 External links Overview ( edit ) The BATNA is often seen by negotiators not as a safety net ,
but rather as a point of leverage in negotiations . Although a negotiator 's alternative options should , in theory , be straightforward to evaluate , the
effort to understand which alternative represents a party 's BATNA is often not invested . Options need to be actual and actionable to be of value , however
without the investment of time , options will frequently be included that fail on one of these criteria . Most managers overestimate their BATNA whilst
simultaneously investing too little time into researching their real options . This can result in poor or faulty decision making and negotiating result .
Negotiators also need to be aware of the other negotiator 's BATNA and to identify how it compares to what they are offering . Some people may adopt aggressive
, coercive , threatening and / or deceptive techniques , this is known as a hard negotiation style ; a theoretical example of this is adversarial approach style
negotiation . Others may employ a soft style , which is friendly , trusting , compromising , and conflict avoiding .
</item 9>

<item 10>
Best alternative to a negotiated agreement
, a non-profit organization . About Wikipedia
</item 10>
=== END ITEM id=ccca502a5e1047a0 ===

=== ITEM id=dc76f8b9a71bc748 ===
Gold sentences and facts:
Sentence 0: Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 ) Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions
  Fact 0.0: The list is titled Ballets by Pyotr Ilyich Tchaikovsky.
  Fact 0.1: Swan Lake, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1876.
  Fact 0.2: Sleeping Beauty, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1889.
  Fact 0.3: The Nutcracker, a ballet by Pyotr Ilyich Tchaikovsky, is dated 1892.
  Fact 0.4: The text contains the label "List of all compositions".

Retrieved items:
<item 1>
The Sleeping Beauty (ballet)
2 in C ♯ minor Grand Piano Sonata in G major Portrayals Tchaikovsky ( 1969 film ) Music competitions International Tchaikovsky Competition International
Tchaikovsky Competition for Young Musicians Related articles Antonina Miliukova Belyayev circle Death Modest Ilyich Tchaikovsky Musical style Nadezhda von Meck
Pyotr Ilyich Tchaikovsky in popular media Symphonies by Pyotr Ilyich Tchaikovsky Tchaikovsky State House - Museum The Five Vladimir Davydov Book : Pyotr Ilyich
Tchaikovsky List of compositions by Pyotr Ilyich Tchaikovsky Category : Pyotr Ilyich Tchaikovsky `` Sleeping Beauty '' / `` Little Briar Rose '' by Charles
Perrault and the Brothers Grimm Theatre La Belle au Bois Dormant ( opera ) The Sleeping Beauty ( ballet ) Variants Sun , Moon ,
</item 1>

<item 2>
The Sleeping Beauty (ballet)
The work has become one of the classical repertoire 's most famous ballets . Contents ( hide ) 1 History 1.1 Composition history 1.2 Performance history 2 Roles
and original cast 3 Instrumentation 4 Roles 5 Synopsis 6 Structure 7 Versions by other hands 7.1 Piano arrangements 7.2 Aurora 's Wedding by Sergei Diaghilev 8
Trademark controversy 9 References 10 External links 10.1 Video samples 10.2 Scores 10.3 History History ( edit ) Composition history ( edit ) Tchaikovsky was
approached by the Director of the Imperial Theatres in St. Petersburg , Ivan Vsevolozhsky on 25 May 1888 about a possible ballet adaptation on the subject of
the story of Undine . It was later decided that Charles Perrault 's La Belle au bois dormant would be the story for which Tchaikovsky would compose the music
for the ballet . Tchaikovsky did not hesitate to accept the commission , although he was aware that his only previous ballet , Swan Lake , met with little
enthusiasm at that stage of his career .
</item 2>

<item 3>
The Sleeping Beauty (ballet)
The Sleeping Beauty ( ballet ) - wikipedia The Sleeping Beauty ( ballet ) Jump to : navigation , search Ballets by Pyotr Ilyich Tchaikovsky Swan Lake ( 1876 )
Sleeping Beauty ( 1889 ) The Nutcracker ( 1892 ) List of all compositions The Sleeping Beauty ( Russian : Спящая красавица / Spyashchaya krasavitsa ) is a
ballet in a prologue and three acts , first performed in 1890 . The music was composed by Pyotr Ilyich Tchaikovsky ( his opus 66 ) . The score was completed in
1889 , and is the second of his three ballets . The original scenario was conceived by Ivan Vsevolozhsky , and is based on Charles Perrault 's La Belle au bois
dormant . The choreographer of the original production was Marius Petipa . The premiere performance took place at the Mariinsky Theatre in St. Petersburg on
January 15 , 1890 . The work has become one of the classical repertoire 's most famous ballets .
</item 3>

<item 4>
The Sleeping Beauty (ballet)
References ( edit ) Jump up ^ Roger Fiske ( 1973 ) : Eulenberg Edition , Foreword to the complete score of the ballet : `` On 2 Feb 1939 Sadler 's Wells
presented the ballet in London with Margot Fonteyn in the title role ... This was the first successful production outside Russia , and it led to The Sleeping
Beauty becoming extremely popular in all countries where classical ballet is cultivated ... The way in which he developed his themes and the lavish originality
with which he scored the music raised his ballets far above those his predecessors had composed . '' Jump up ^ `` The Sleeping Beauty '' . Collections Online .
Royal Opera House . Retrieved 9 April 2016 . Jump up ^ `` Review / Ballet ; San Francisco ' Sleeping Beauty ' Underlines Russian Influences '' . Retrieved
November 27 , 2013 . New York Times / Anna Kisselgoff , March 20 , 1990 ^ Jump up to : Brillarelli , Livia ( 1995 ) . Cecchetti A Ballet Dynasty . Toronto :
Dance Collection Danse Educational Publications . p. 31 .
</item 4>

<item 5>
The Sleeping Beauty (ballet)
* 1895 ) The Sleeping Beauty ( 1890 ) Nénuphar ( 1890 ) Kalkabrino ( 1891 ) A Fairy Tale ( 1891 ) La Sylphide ( * 1892 ) The Nutcracker ( 1892 ) Cinderella (
1893 ) The Awakening of Flora ( 1894 ) Swan Lake ( * 1895 ) The Little Humpbacked Horse ( as `` The Tsar Maiden '' ) ( * 1895 ) The Calvary Halt ( 1896 ) The
Pearl ( 1896 ) Bluebeard ( 1896 ) Raymonda ( 1898 ) 1900 -- 03 Les Ruses d'Amour or The Trial of Damis ( 1900 ) The Seasons ( 1900 ) Harlequinade ( 1900 ) The
Heart of the Marquis ( 1902 ) The Magic Mirror ( 1903 ) The Romance of the Rosebud and the Butterfly ( never presented ) Pyotr Ilyich Tchaikovsky Operas The
Voyevoda Undina The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa
</item 5>

<item 6>
The Sleeping Beauty (ballet)
19 , as no. 18 was cut in the original production No. 18 Entr'acte symphonique ( solo for violin with light orchestral accompaniment , composed for Leopold Auer
, cut from the original production ) Scene II -- Le château de la belle au bois dormant No. 19 Scène du château de sommeil No. 20 Scène et final -- Le réveil
d'Aurore The Bluebird and Princess Florina ( Valeri Panov and Natalia Makarova ) from the 1964 Russian motion picture featuring artists of the Kirov Ballet .
Act III -- Les Noces de Désiré et d'Aurore No. 21 Marche No. 22 Grand polonaise dansée ( a.k.a. The Procession of the Fairy Tales ) Grand divertissement -- No.
23 Pas de quatre a . Entrée b . Variation de la fée - Or ( transferred by Petipa to Act II as a variation for Carlotta Brianza in the original production ) c .
</item 6>

<item 7>
The Sleeping Beauty (ballet)
This version has been performed by a number of central European ballet companies over the past two decades . Roles and original cast ( edit ) Role Mariinsky
1890 Bolshoi 1899 King Florestan Feliks Krzesiński ( Mathilde Kschessinska 's father ) Queen Giuseppina Cecchetti Princess Aurora , the Sleeping Beauty Carlotta
Brianza Lyubov Roslavleva Lilac Fairy Marie Petipa M. Grachevskaya Carabosse Enrico Cecchetti Vasiliy Geltser Prince Désiré Pavel Gerdt Ivan Khlyustin Bluebird
Enrico Cecchetti Princess Florine Varvara Nikitina The ballet 's premiere received more favorable accolades than Swan Lake from the press but Tchaikovsky never
had the luxury of being able to witness his work become an instant success in theatres outside of Russia . He died in 1893 .
</item 7>

<item 8>
The Sleeping Beauty (ballet)
The ballet scenario that Tchaikovsky worked on was based on the Brothers Grimm 's version of Perrault 's work entitled ' Dornröschen ' . In that version , the
Princess 's parents ( the King and the Queen ) survived the 100 - year sleep to celebrate the Princess 's wedding to the Prince . However , Vsevolozhsky
incorporated Perrault 's other characters from his stories into the ballet , such as Puss in Boots , Little Red Riding Hood , Cinderella , Bluebird , Bluebeard
, Ricky of the Tuft and Tom Thumb . Other French fairy tale characters to be featured are Beauty and the Beast , Pretty Goldilocks and The White Cat .
Regardless , Tchaikovsky was happy to inform the Director of the Imperial Theatre that he had great pleasure studying the work and had come away with adequate
inspiration to do it justice . The choreographer was Marius Petipa , ballet master of the Imperial Ballet , who wrote a very detailed list of instructions as to
the musical requirements .
</item 8>

<item 9>
The Sleeping Beauty (ballet)
With subtitles in Esperanto . Solo of Prince Désiré from Pas de deux ( Act 3 ) danced by Rudolf Nureyev ( from YouTube ) Scores ( edit ) Sleeping Beauty :
Scores at the International Music Score Library Project ( IMSLP ) History ( edit ) Tchaikovsky Research Ballets by Pyotr Ilyich Tchaikovsky Swan Lake , Op. 20
Sleeping Beauty , Op. 66 The Nutcracker , Op. 71 List of compositions by Pyotr Ilyich Tchaikovsky The ballets and * revivals of Marius Petipa in Russia 1847 --
59 Paquita ( * 1847 , * 1881 ) Le Diable amoureux ( as `` Satanella '' ) ( * 1848 ) Leda , the Swiss Milkmaid ( 1849 ) Giselle ( * 1850 , * 1884 , * 1899 , *
1903 ) The Star of Granada ( 1855 ) The Rose , the Violet , and the Butterfly ( 1857 ) Le Corsaire ( * 1858 , * 1863 , * 1868 ,
</item 9>

<item 10>
The Sleeping Beauty (ballet)
The Oprichnik Vakula the Smith Eugene Onegin The Maid of Orleans Mazepa ( or Mazeppa ) Cherevichki The Enchantress The Queen of Spades Iolanta Ballets Swan Lake
The Sleeping Beauty The Nutcracker Symphonies No. 1 in G minor ( `` Winter Daydreams '' ) No. 2 in C minor ( `` Little Russian '' ) No. 3 in D major ( `` Polish
'' ) No. 4 in F minor Symphony in B minor ( `` Manfred '' ) No. 5 in E minor No. 6 in B minor ( `` Pathétique '' ) No. 7 in E ♭ major ( unfinished ) Concertante
Piano Concerto No. 1 in B ♭ minor Piano Concerto No. 2 in G major Piano Concerto No.
</item 10>
=== END ITEM id=dc76f8b9a71bc748 ===

=== ITEM id=ff8458deedf5eb0f ===
Gold sentences and facts:
Sentence 0: The original start date was January 2002 , but was pushed to February 7 in Los Angeles , California . Locations included Burbank , Downey , New York City , LA / Ontario International Airport ( which doubled for Miami International Airport ) , Quebec City and Montreal . The film was shot in 147 locations in only 52 days . DiCaprio reflected , `` Scenes that we thought would take three days took an afternoon '' . Filming ran from April 25 -- 30 in Park Avenue , just outside the Waldorf - Astoria Hotel . Production moved to Orange , New Jersey and returned to Brooklyn for bank and courthouse scenes . Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport . Quebec City was chosen for its European character and French feel . Place Royale , within Old Quebec , stands for Montrichard -- the church in the background of the arrest scene is Notre - Dame - des - Victoires . Filming ended on May 12 in Montreal .
  Fact 0.0: The original start date was January 2002.
  Fact 0.1: The start date was pushed to February 7 in Los Angeles, California.
  Fact 0.2: Locations included Burbank.
  Fact 0.3: Locations included Downey.
  Fact 0.4: Locations included New York City.
  Fact 0.5: Locations included LA / Ontario International Airport.
  Fact 0.6: LA / Ontario International Airport doubled for Miami International Airport.
  Fact 0.7: Locations included Quebec City.
  Fact 0.8: Locations included Montreal.
  Fact 0.9: The film was shot in 147 locations.
  Fact 0.10: The film was shot in only 52 days.
  Fact 0.11: DiCaprio reflected that scenes thought to take three days took an afternoon.
  Fact 0.12: Filming ran from April 25 to 30 in Park Avenue.
  Fact 0.13: Park Avenue filming was just outside the Waldorf-Astoria Hotel.
  Fact 0.14: Production moved to Orange, New Jersey.
  Fact 0.15: Production returned to Brooklyn for bank and courthouse scenes.
  Fact 0.16: Shooting also took place at the TWA Flight Center at John F. Kennedy International Airport.
  Fact 0.17: Quebec City was chosen for its European character.
  Fact 0.18: Quebec City was chosen for its French feel.
  Fact 0.19: Place Royale, within Old Quebec, stands for Montrichard.
  Fact 0.20: The church in the background of the arrest scene is Notre-Dame-des-Victoires.
  Fact 0.21: Filming ended on May 12 in Montreal.
Sentence 1: Catch Me If You Can is a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson . The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen , and Nathalie Baye in supporting roles .
  Fact 1.0: Catch Me If You Can is a 2002 American biographical crime film.
  Fact 1.1: Catch Me If You Can was directed by Steven Spielberg.
  Fact 1.2: Catch Me If You Can was co-produced by Steven Spielberg.
  Fact 1.3: Catch Me If You Can has a screenplay by Jeff Nathanson.
  Fact 1.4: The film is based on the life of Frank Abagnale.
  Fact 1.5: Frank Abagnale successfully performed cons worth millions of dollars before his 19th birthday.
  Fact 1.6: Frank Abagnale performed cons by posing as a Pan American World Airways pilot.
  Fact 1.7: Frank Abagnale performed cons by posing as a Georgia doctor.
  Fact 1.8: Frank Abagnale performed cons by posing as a Louisiana parish prosecutor.
  Fact 1.9: Abagnale's primary crime was check fraud.
  Fact 1.10: Abagnale became so experienced that the FBI eventually turned to him for help.
  Fact 1.11: The FBI turned to Abagnale for help in catching other checking forgers.
  Fact 1.12: The film stars Leonardo DiCaprio.
  Fact 1.13: The film stars Tom Hanks.
  Fact 1.14: Christopher Walken, Martin Sheen, and Nathalie Baye have supporting roles in the film.
Sentence 2: Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received positively by both critics and Abagnale .
  Fact 2.0: Development for the film started in 1980.
  Fact 2.1: Development did not progress until 1997.
  Fact 2.2: In 1997, the film rights to Abagnale's book were sold to Spielberg's DreamWorks.
  Fact 2.3: David Fincher was a possible candidate for director before Spielberg decided to direct.
  Fact 2.4: Gore Verbinski was a possible candidate for director before Spielberg decided to direct.
  Fact 2.5: Lasse Hallström was a possible candidate for director before Spielberg decided to direct.
  Fact 2.6: Miloš Forman was a possible candidate for director before Spielberg decided to direct.
  Fact 2.7: Cameron Crowe was a possible candidate for director before Spielberg decided to direct.
  Fact 2.8: Filming took place from February to May 2002.
  Fact 2.9: The film was a financial success.
  Fact 2.10: The film was a critical success.
  Fact 2.11: The film was received positively by both critics and Abagnale.

Retrieved items:
<item 1>
*Catch Me If You Can* is a 2002 American biographical crime film directed and co-produced by Steven Spielberg, with a screenplay by Jeff Nathanson based on
Frank Abagnale Jr. and Stan Redding’s book. Starring Leonardo DiCaprio and Tom Hanks, it was marketed as “inspired by a true story.” Released December 25, it
earned $352.1 million worldwide on a $52 million budget, becoming 2002’s 11th-highest
</item 1>

<item 2>
*Catch Me If You Can* (2002), directed by Steven Spielberg and starring Leonardo DiCaprio and Tom Hanks, follows teenage Frank Abagnale Jr.’s frauds and FBI
agent Carl Hanratty’s pursuit. Inspired by Abagnale’s memoir, it altered events, including inventing a continuing father-son bond. Filmed across 147 locations
in 52 days, it grossed $352.1 million on a $52 million budget and earned generally positive reviews.
</item 2>

<item 3>
Catch Me If You Can
The film is based on the life of Frank Abagnale , who , before his 19th birthday , successfully performed cons worth millions of dollars by posing as a Pan
American World Airways pilot , a Georgia doctor and a Louisiana parish prosecutor . His primary crime was check fraud ; he became so experienced that the FBI
eventually turned to him for help in catching other checking forgers . The film stars Leonardo DiCaprio and Tom Hanks , with Christopher Walken , Martin Sheen ,
and Nathalie Baye in supporting roles . Development for the film started in 1980 , but did not progress until 1997 , when the film rights to Abagnale 's book
were sold to Spielberg 's DreamWorks . David Fincher , Gore Verbinski , Lasse Hallström , Miloš Forman , and Cameron Crowe had all been possible candidates for
director before Spielberg decided to direct . Filming took place from February to May 2002 . The film was a financial and critical success , and was received
positively by both critics and Abagnale .
</item 3>

<item 4>
The references document the development and production of *Catch Me If You Can* from 1997 to 2002. Variety and IGN reports cover script work, Leonardo
DiCaprio’s casting and scheduling, director changes—including Lasse Hallström taking over—and other production and publicity updates. The list also cites
production notes and 2002 entertainment coverage.
</item 4>

<item 5>
Catch Me If You Can
British Board of Film Classification . December 13 , 2002 . Retrieved February 13 , 2016 . Jump up ^ Van Luling , Todd ( October 17 , 2014 ) . `` 11 Easter Eggs
You Never Noticed In Your Favorite Movies '' . The Huffington Post . TheHuffingtonPost.com , Inc . Retrieved January 29 , 2015 . ^ Jump up to : Frank Abagnale (
September 3 , 2002 ) . `` Comments '' . Abagnale & Associates . Archived from the original on February 16 , 2009 . Retrieved June 30 , 2008 . ^ Jump up to :
Claude Brodesser ; Dana Harris ( August 21 , 2001 ) . `` D'Works to play ' Catch ' '' . Variety . Retrieved June 30 , 2008 . Jump up ^ Charles Lyons ; Dade
Hayes ( January 8 , 2002 ) . `` D'Works sets play date for pricey ' Catch ' '' . Variety . Retrieved June 29 , 2008 .
</item 5>

<item 6>
Catch Me If You Can
Catch Me If You Can - wikipedia Catch Me If You Can Jump to : navigation , search This article is about the 2002 film . For other uses , see Catch Me If You Can
( disambiguation ) . Catch Me If You Can Theatrical release poster Directed by Steven Spielberg Produced by Steven Spielberg Walter F. Parkes Screenplay by Jeff
Nathanson Based on Catch Me If You Can by Frank Abagnale and Stan Redding Starring Leonardo DiCaprio Tom Hanks Music by John Williams Cinematography Janusz
Kamiński Edited by Michael Kahn Production company Amblin Entertainment Parkes / MacDonald Productions Distributed by DreamWorks Pictures Release date December
25 , 2002 ( 2002 - 12 - 25 ) Running time 141 minutes Country United States Language English Budget $52 million Box office $352.1 million Catch Me If You Can is
a 2002 American biographical crime film directed and co-produced by Steven Spielberg from a screenplay by Jeff Nathanson .
</item 6>

<item 7>
Catch Me If You Can
Williams also earned a Grammy Award nomination . Elements of the film were later parodied in The Simpsons episode `` Catch ' Em If You Can '' . Home media (
edit ) Catch Me If You Can was released on DVD on May 6 , 2003 and on Blu - ray on December 4 , 2012 . Musical adaptation ( edit ) Main article : Catch Me If
You Can ( musical ) A musical adaptation of the same name premiered at the 5th Avenue Theatre in Seattle , Washington in July 2009 , starring Aaron Tveit and
Norbert Leo Butz . It began previews on Broadway at the Neil Simon Theatre on March 11 , 2011 and officially opened April 10 , 2011 . The musical was nominated
for four Tony Awards , including Best Musical . See also ( edit ) The Great Impostor , a 1961 film based on the story of an impostor named Ferdinand Waldo
Demara . The Pretender , a TV series References ( edit ) Jump up ^ `` CATCH ME IF YOU CAN ( 12A ) '' . British Board of Film Classification . December 13 , 2002
.
</item 7>

<item 8>
Catch Me If You Can
The marketing department was careful to market the film as `` inspired by a true story '' in order to avoid controversy similar to that surrounding A Beautiful
Mind ( 2001 ) and The Hurricane ( 1999 ) , both of which deviated from history . The premiere took place at Westwood , Los Angeles , California on December 18 ,
2002 . Box Office ( edit ) Catch Me If You Can was released on December 25 , 2002 , earning slightly above $30 million in 3,225 theaters during its opening
weekend . The film went on to gross $164.6 million in North America and $187.5 million in foreign countries , coming at a worldwide total of $352.1 million .
The film was a financial success , recouping the $52 million budget six times over . Catch Me If You Can was the eleventh highest - grossing film of 2002 .
Minority Report ( also directed by Spielberg ) was tenth highest . Critical response ( edit ) On Rotten Tomatoes , the film has a certified fresh rating of 96 %
, based on 197 reviews , with an average rating of 7.9 / 10 .
</item 8>

<item 9>
*Catch Me If You Can* tells the story of Frank Abagnale, who, before age 19, stole millions through check fraud while impersonating an airline pilot, doctor,
and prosecutor. The FBI later recruited him to help catch forgers. Directed by Steven Spielberg and starring Leonardo DiCaprio and Tom Hanks, the film was shot
in 2002 and became a critical and financial success. Critics praised its performances, style, brisk pace, and entertaining tone; Metacritic rated
</item 9>

<item 10>
Catch Me If You Can
wikipedia.org/w/index.php?title=Catch_Me_If_You_Can&oldid=821436887 '' Categories : 2002 films English - language films 2000s biographical films 2000s crime
drama films Amblin Entertainment films American biographical films American chase films American crime drama films American films Biographical films about
fraudsters Crime films based on actual events Drama films based on actual events DreamWorks Pictures films Films scored by John Williams Films about con artists
Films about identity theft Films based on biographies Films directed by Steven Spielberg Films set in Atlanta Films set in California Films set in Florida Films
set in France Films set in Louisiana Films set in Westchester County , New York Films set in New York City Films set in the 1960s Films shot in California Films
shot in Montreal Films produced by Steven Spielberg True crime films Fraud in fiction Hidden categories : Webarchive template wayback links Use mdy dates from
October 2015 Good articles Wikipedia articles with GND identifiers Talk Contents About Wikipedia Wikiquote বাংলা Български Català Čeština Dansk Deutsch Español
Euskara فارسی Français Galego 한국어
</item 10>
=== END ITEM id=ff8458deedf5eb0f ===

=== ITEM id=a0e023206a6c3670 ===
Gold sentences and facts:
Sentence 0: Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
  Fact 0.0: Uncle Tom's Cabin was serialized in The National Era for ten months.
  Fact 0.1: The serialization of Uncle Tom's Cabin in The National Era began with the June 5, 1851 issue.
  Fact 0.2: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852.
  Fact 0.3: The book edition of Uncle Tom's Cabin was published by John P. Jewett and Company.
Sentence 1: Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical , starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story 's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's Cabin in book form , she eventually consented to the request .
  Fact 1.0: Uncle Tom's Cabin first appeared as a 40-week serial in The National Era.
  Fact 1.1: The National Era was an abolitionist periodical.
  Fact 1.2: The serial started with the June 5, 1851 issue.
  Fact 1.3: Uncle Tom's Cabin was originally intended as a shorter narrative that would run for only a few weeks.
  Fact 1.4: Harriet Beecher Stowe expanded the story significantly.
  Fact 1.5: The expanded story was instantly popular.
  Fact 1.6: Several protests were sent to the National Era office when Stowe missed an issue.
  Fact 1.7: The publisher John P. Jewett contacted Stowe about turning the serial into a book because of the story's popularity.
  Fact 1.8: Stowe questioned whether anyone would read Uncle Tom's Cabin in book form.
  Fact 1.9: Stowe eventually consented to the request to publish the story as a book.
Sentence 2: Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or , Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company . Followed by A Key to Uncle Tom 's Cabin ( 1853 )
  Fact 2.0: The infobox author of Uncle Tom's Cabin is Harriet Beecher Stowe.
  Fact 2.1: The original title of Uncle Tom's Cabin is Uncle Tom's Cabin; or, Life Among the Lowly.
  Fact 2.2: The illustrator of the 1st edition of Uncle Tom's Cabin is Hammatt Billings.
  Fact 2.3: Uncle Tom's Cabin is from the United States.
  Fact 2.4: Uncle Tom's Cabin is written in English.
  Fact 2.5: Uncle Tom's Cabin is a novel.
  Fact 2.6: Uncle Tom's Cabin was serialized in The National Era for ten months, beginning with the June 5, 1851 issue.
  Fact 2.7: Uncle Tom's Cabin was published as a book in two volumes on March 20, 1852, by John P. Jewett and Company.
  Fact 2.8: Uncle Tom's Cabin is followed by A Key to Uncle Tom's Cabin (1853).
  Fact 2.9: The infobox includes an image of the Boston edition of Uncle Tom's Cabin.

Retrieved items:
<item 1>
Uncle Tom's Cabin
Convinced the book would be popular , Jewett made the unusual decision ( for the time ) to have six full - page illustrations by Hammatt Billings engraved for
the first printing . Published in book form on March 20 , 1852 , the novel sold 3,000 copies on that day alone , and soon sold out its complete print run . A
number of other editions were soon printed ( including a deluxe edition in 1853 , featuring 117 illustrations by Billings ) . In the first year of publication ,
300,000 copies of Uncle Tom 's Cabin were sold . At that point , however , `` demand came to an unexpected halt ... No more copies were produced for many years
, and if , as is claimed , Abraham Lincoln greeted Stowe in 1862 as ' the little woman who wrote the book that made this great war , ' the work had effectively
been out of print for many years . '' Jewett went out of business , and it was not until Ticknor and Fields put the work back in print in November 1862 that
demand began again to increase .
</item 1>

<item 2>
Uncle Tom's Cabin
Uncle Tom 's Cabin - wikipedia Uncle Tom 's Cabin Jump to : navigation , search This article is about the mid-19th century novel . For other uses , see Uncle
Tom 's Cabin ( disambiguation ) . Uncle Tom 's Cabin Uncle Tom 's Cabin , Boston edition Author Harriet Beecher Stowe Original title Uncle Tom 's Cabin ; or ,
Life Among the Lowly Illustrator Hammatt Billings ( 1st edition ) Country United States Language English Genre Novel Published Serialized in The National Era
for ten months , beginning with the June 5 , 1851 issue . Published as a book in two volumes on March 20 , 1852 , John P. Jewett and Company .
</item 2>

<item 3>
Uncle Tom's Cabin
Stowe mentioned a number of the inspirations and sources for her novel in A Key to Uncle Tom 's Cabin ( 1853 ) . This non-fiction book was intended to verify
Stowe 's claims about slavery . However , later research indicated that Stowe did not read many of the book 's cited works until after she had published her
novel . Publication The National Era , June 5 , 1851 Uncle Tom 's Cabin first appeared as a 40 - week serial in The National Era , an abolitionist periodical ,
starting with the June 5 , 1851 , issue . It was originally intended as a shorter narrative that would run for only a few weeks . Stowe expanded the story
significantly , however , and it was instantly popular , such that several protests were sent to the Era office when she missed an issue . Because of the story
's popularity , the publisher John P. Jewett contacted Stowe about turning the serial into a book . While Stowe questioned if anyone would read Uncle Tom 's
Cabin in book form , she eventually consented to the request .
</item 3>

<item 4>
Harriet Beecher Stowe’s anti-slavery novel *Uncle Tom’s Cabin* (1852), first published in serial form, provoked fierce opposition in the American South. William
Gilmore Simms called it false; slavery supporters denounced it as criminal and slanderous. A Mobile bookseller was driven out for selling it, and Stowe received
threats, including a package containing a severed enslaved person’s ear. Southern writers responded with books opposing the novel. Related works include *The
Octoroon
</item 4>

<item 5>
Harriet Beecher Stowe, a Connecticut teacher and abolitionist, wrote *Uncle Tom’s Cabin*, a sentimental novel portraying slavery’s brutality and the power of
Christian love. Centered on the long-suffering enslaved Tom, it became the 19th century’s best-selling novel and second-best-selling book after the Bible.
First-year sales reached 300,000 in the U.S. and one million in Britain. The novel fueled abolitionism, provoked pro-slavery rebuttals,
</item 5>

<item 6>
Uncle Tom's Cabin
The book was translated into all major languages , and in the United States it became the second best - selling book after the Bible . A number of the early
editions carried an introduction by Rev James Sherman , a Congregational minister in London noted for his abolitionist views . Uncle Tom 's Cabin sold equally
well in Britain , with the first London edition appearing in May 1852 and selling 200,000 copies . In a few years over 1.5 million copies of the book were in
circulation in Britain , although most of these were infringing copies ( a similar situation occurred in the United States ) . Plot Eliza escapes with her son ,
Tom sold `` down the River '' Full - page illustration by Hammatt Billings for Uncle Tom 's Cabin depicts Eliza telling Uncle Tom that he has been sold and she
is running away to save her child ( first edition : Boston : John P. Jewett and Company , 1852 ) . The book opens with a Kentucky farmer named Arthur Shelby
facing the loss of his farm because of debts .
</item 6>

<item 7>
*Uncle Tom’s Cabin* is an American sentimental novel about slavery, first published in serial form in American magazines, including political magazines. It has
been adapted into comics, films, plays, and television programs. The supplied text also consists largely of Wikipedia navigation, source, and page-protection
metadata.
</item 7>

<item 8>
Stowe’s *Uncle Tom’s Cabin* began as a 40-week serial in the abolitionist *National Era* on June 5, 1851. Its unexpected popularity led publisher John P. Jewett
to issue it as a book. In 1853, Stowe published *A Key to Uncle Tom’s Cabin* to support her claims about slavery, though later research found she read many
cited sources only after writing the novel. The *Key* also condemned laws that enabled abuse
</item 8>

<item 9>
Jewett published *Uncle Tom’s Cabin* on March 20, 1852, with six engraved illustrations—a novelty for the time. It sold 3,000 copies on its first day and
300,000 in its first year; an 1853 deluxe edition included 117 Billings illustrations. Demand then unexpectedly collapsed, and the novel remained out of print
for years as Jewett went out of business. Ticknor and Fields reissued it in November 1862, rev
</item 9>

<item 10>
Harriet Beecher Stowe’s *Uncle Tom’s Cabin* became the United States’ second-best-selling book after the Bible and sold widely in Britain, where the 1852 first
edition sold 200,000 copies. Translated into nearly every major language—including Chinese in 1901 and Amharic in 1930—it inspired debate and reportedly
influenced readers’ perceptions of slavery. The novel opens with indebted Kentucky farmer Arthur Shelby; Eliza flees with her son while Tom is sold
</item 10>
=== END ITEM id=a0e023206a6c3670 ===

=== ITEM id=9daa20ce9e05d886 ===
Gold sentences and facts:
Sentence 0: The poem survives in a single manuscript , the Cotton Nero A.x. , which also includes three religious narrative poems : Pearl , Purity and Patience . All are thought to have been written by the same unknown author , dubbed the `` Pearl Poet '' or `` Gawain Poet '' , since all four are written in a North West Midland dialect of Middle English .
  Fact 0.0: The poem survives in a single manuscript, the Cotton Nero A.x.
  Fact 0.1: The Cotton Nero A.x. manuscript also includes three religious narrative poems: Pearl, Purity and Patience.
  Fact 0.2: Pearl, Purity and Patience are thought to have been written by the same unknown author.
  Fact 0.3: The unknown author is dubbed the 'Pearl Poet' or 'Gawain Poet'.
  Fact 0.4: The 'Pearl Poet' or 'Gawain Poet' name is given because all four poems are written in a North West Midland dialect of Middle English.
  Fact 0.5: All four poems are written in a North West Midland dialect of Middle English.

Retrieved items:
<item 1>
Sir Gawain and the Green Knight
6 Homoerotic interpretations 8 Modern adaptations 8.1 Books 8.2 Film and television 8.3 Theatre 8.4 Opera 9 References 10 External links Synopsis ( edit ) Sir
Gawain and the Green Knight ( from original manuscript , artist unknown ) In Camelot on New Year 's Day , King Arthur 's court is exchanging gifts and waiting
for the feasting to start when the king asks first to see or hear of an exciting adventure . A gigantic figure , entirely green in appearance and riding a green
horse , rides unexpectedly into the hall . He wears no armour but bears an axe in one hand and a holly bough in the other . Refusing to fight anyone there on
the grounds that they are all too weak to take him on , he insists he has come for a friendly `` Christmas game '' : someone is to strike him once with his axe
on condition that the Green Knight may return the blow in a year and a day . The splendid axe will belong to whoever takes him on .
</item 1>

<item 2>
Sir Gawain and the Green Knight
Other modern translations include those by Larry Benson , Brian Stone , James Winny , Helen Cooper , W.S. Merwin , Jacob Rosenberg , William Vantuono , Joseph
Glaser , Bernard O'Donoghue , John Gardner , and Francis Ingledew . A new and illustrated translation of the work by illustrator and historian Michael Smith is
currently being crowdsourced by publisher Unbound with a planned publication date of 2018 . Sword of the Valiant ( 1984 ) , one of two film adaptations ,
starring Sean Connery as the Green Knight Film and television ( edit ) The poem has been adapted to film twice , on both occasions by writer - director Stephen
Weeks : first as Gawain and the Green Knight in 1973 and again in 1984 as Sword of the Valiant : The Legend of Sir Gawain and the Green Knight , featuring Miles
O'Keeffe as Gawain and Sean Connery as the Green Knight . Both films have been criticised for deviating from the poem 's plot . Also , Bertilak and the Green
Knight are never connected .
</item 2>

<item 3>
*Sir Gawain and the Green Knight* is a medieval chivalric romance preserved in the Cotton Nero A.x. manuscript alongside *Pearl*, *Purity*, and *Patience*. At
Arthur’s New Year feast, Gawain accepts a beheading challenge: he strikes the mysterious Green Knight, then must receive a return blow a year and a day later.
Tested at Lord Bertilak’s castle by its lady, Gawain struggles to uphold chivalry, loyalty, and honour.
</item 3>

<item 4>
The references document scholarship and adaptations of *Sir Gawain and the Green Knight*. Studies examine the poem’s “year and a day” convention, hunting and
deer, the Green Knight’s connection to nature, seasonal and apocalyptic imagery, symbolism of green and gold, gifts, and games. The list also includes editions,
translations, reviews, a BBC program, theatrical productions, and musical works, including operas by Richard Blackford and others.
</item 4>

<item 5>
Sir Gawain and the Green Knight
Retrieved 2016 - 03 - 01 . External links ( edit ) Find more aboutSir Gawain and the Green Knightat Wikipedia 's sister projects Definitions from Wiktionary
Media from Wikimedia Commons News from Wikinews Quotations from Wikiquote Texts from Wikisource Learning resources from Wikiversity Online texts High -
resolution , full - sized scan of entire manuscript Sir Gawain and the Green Knight at Archive.org JRR Tolkien and EV Gordon 's Edition The poem in Middle
English Sir Gawain and the Green Knight public domain audiobook at LibriVox ( various translations ) Google Books results General information The Gawain / Pearl
Poet from the University of Calgary The Camelot Project Info on Sir Gawain Luminarium SGGK Website Oxford Bibliographies : Bibliography on Sir Gawain and the
Green Knight BBC documentary tour of locations thought to be found in the poem ... Holywell , the River Mersey , the ' Wild Wirral ' , the Peak District , the
Roaches , and Lud Chapel Sir Gawain Introduction - Article introducing various translations and adaptations of Sir Gawain The Pearl Poet Works Sir Gawain and
the Green Knight Pearl Patience Cleanness St.
</item 5>

<item 6>
Sir Gawain and the Green Knight
`` King Arthur and King Cornwall '' . In Sir Gawain : Eleven Romances and Tales . Western Michigan University Medieval Institute Publications , 2000 . p. 427 .
ISBN 1 - 879288 - 59 - 1 ^ Jump up to : Lasater , Alice E. Spain to England : A Comparative Study of Arabic , European , and English Literature of the Middle
Ages . University Press of Mississippi , 1974 . ISBN 0 - 87805 - 056 - 6 Jump up ^ Rix , Michael M. `` A Re-Examination of the Castleton Garlanding '' .
Folklore . ( June 1953 ) 64.2 pp. 342 -- 344 Jump up ^ Buchanan , Alice ( June 1932 ) . `` The Irish Framework of Sir Gawain and the Green Knight '' . PLMA. 47
( 2 ) : 315 -- 338 . doi : 10.2307 / 457878 . Jump up ^ . Friedman , Albert B. Morgan le Fay in Sir Gawain and the Green Knight . Medieval Academy of America .
</item 6>

<item 7>
Sir Gawain and the Green Knight
Sir Gawain and the Green Knight - wikipedia Sir Gawain and the Green Knight Jump to : navigation , search For the 1973 film adaptation , see Gawain and the
Green Knight ( film ) . Sir Gawain and the Green Knight First page of survived manuscript , circa 14th century . Author Anonymous Country Kingdom of England
Language Middle English Genre Poem , chivalric romance , Arthurian and alliterative verse . Publication date 14th century Sir Gawain and the Green Knight (
Middle English : Sir Gawayn and þe Grene Knyȝt ) is a late 14th - century Middle English chivalric romance . It is one of the best known Arthurian stories ,
with its plot combining two types of folklore motifs , the beheading game and the exchange of winnings . The Green Knight is interpreted by some as a
representation of the Green Man of folklore and by others as an allusion to Christ . Written in stanzas of alliterative verse , each of which ends in a rhyming
bob and wheel , it draws on Welsh , Irish , and English stories , as well as the French chivalric tradition .
</item 7>

<item 8>
The references document extensive scholarship on *Sir Gawain and the Green Knight*: editions, translations, sources and analogues, and studies of its structure,
symbolism, hunting scenes, virtue, Christian and Roman traditions, gender, courtly love, and Morgan le Fay. They also cover feminist, queer, and postcolonial
readings, textual performance, the poem’s landscape and locations, and related Gawain romances. Cited works include studies by Kittredge, Burrow, Tolkien, and
numerous
</item 8>

<item 9>
*Sir Gawain and the Green Knight* is an anonymous late-14th-century Middle English Arthurian romance, probably by the “Pearl Poet,” whose identity remains
uncertain. Its 2,530 alliterative lines combine the beheading game and exchange-of-winnings motifs, drawing on British folklore and French chivalric tradition.
The Green Knight may symbolize the Green Man or Christ. Themes include temptation, honesty, nature, and chivalry. After Gawain’s adventure
</item 9>

<item 10>
*Sir Gawain and the Green Knight*, an anonymous late-14th-century alliterative Arthurian romance preserved in the Cotton Nero A.x manuscript, follows Gawain’s
beheading challenge and tests at Bertilak’s castle. He conceals a protective green girdle, earning only a light wound but exposing fear and imperfect honor. The
poem explores chivalry, faith, temptation, nature, gender, and cultural identity; its pentangle, colors, games, and seasons invite varied
</item 10>
=== END ITEM id=9daa20ce9e05d886 ===

=== ITEM id=dfdcf8a5a10acae1 ===
Gold sentences and facts:
Sentence 0: Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
  Fact 0.0: The glenoid cavity is relatively shallow compared to the acetabulum, which is at the hip joint.
  Fact 0.1: The shallow glenoid cavity makes the shoulder joint prone to dislocation (luxation).
  Fact 0.2: Strong glenohumeral ligaments and muscles prevent dislocation in most cases.

Retrieved items:
<item 1>
Glenoid cavity
ISBN 1 - 4051 - 3413 - 5 . This article incorporates text in the public domain from the 20th edition of Gray 's Anatomy ( 1918 ) References ( edit ) Larson ,
Susan G. ( 2009 ) . `` Evolution of the Hominin Shoulder : Early Homo '' . In Grine , Frederick E. ; Fleagle , John G. ; Leakey , Richard E. The First Humans --
Origin and Early Evolution of the Genus Homo . Springer . doi : 10.1007 / 978 - 1 - 4020 - 9980 - 9 . ISBN 978 - 1 - 4020 - 9979 - 3 . ANATOMY & PHYSIOLOGY :
THE UNITY OF FORM AND FUNCTION , SIXTH EDITION Published by McGraw - Hill Written by Kenneth Saladin
http://www.ucsfhealth.org/conditions/glenoid_labrum_tear/index.html External links ( edit ) Wikimedia Commons has media related to Glenoid cavity . Diagram at
cerrocoso.edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online ,
</item 1>

<item 2>
Glenoid cavity
edu Anatomy figure : 03 : 02 - 07 at Human Anatomy Online , SUNY Downstate Medical Center Mechanics of Glenohumeral Instability at University of Washington
Department of Orthopaedics ( hide ) Bones of the arm Shoulder girdle , clavicle conoid tubercle trapezoid line costal tuberosity subclavian groove Scapula
fossae ( subscapular , supraspinatous , infraspinatous ) notches ( suprascapular , great scapular ) glenoid cavity tubercles ( infraglenoid , supraglenoid )
spine of scapula acromion coracoid process angles ( superior , inferior , lateral ) Humerus upper extremity : necks ( anatomical , surgical ) tubercles (
greater , lesser ) bicipital groove body : radial sulcus deltoid tuberosity lower extremity : capitulum trochlea epicondyles ( lateral , medial ) supracondylar
ridges ( lateral , medial ) fossae ( radial , coronoid , olecranon ) Forearm Radius near elbow ( head , tuberosity ) near wrist ( ulnar notch , styloid process
,
</item 2>

<item 3>
Middle cranial fossa
It is traversed by the squamosal , sphenoparietal , sphenosquamosal , and sphenopetrosal sutures . It houses the temporal lobes of the brain and the pituitary
gland . A middle fossa craniotomy is one means to surgically remove acoustic neuromas ( vestibular schwannoma ) growing within the internal auditory canal of
the temporal bone . Middle cranial fossa surgical anatomy as demonstrated in a right cadaver temporal bone by Dr Jack M Kartush - view from above . Contents (
hide ) 1 Middle part 2 Lateral parts 3 See also 4 References 5 Additional images 6 External links Middle part ( edit ) The middle part of the fossa presents ,
in front , the chiasmatic groove and tuberculum sellae ; the chiasmatic groove ends on either side at the optic foramen , which transmits the optic nerve and
ophthalmic artery to the orbital cavity . Behind the optic foramen the anterior clinoid process is directed backward and medialward and gives attachment to the
tentorium cerebelli .
</item 3>

<item 4>
Glenoid cavity
tuberosity ) near wrist ( ulnar notch , styloid process , Lister 's tubercle ) Ulna near elbow ( tuberosity , olecranon , coronoid process , radial notch ,
trochlear notch ) near wrist ( styloid process ) Hand Carpal bones scaphoid lunate triquetral pisiform trapezium trapezoid capitate hamate hamulus Metacarpal
bones 1st 2nd 3rd 4th 5th Phalanges proximal intermediate distal Retrieved from `` https://en.wikipedia.org/w/index.php?title=Glenoid_cavity&oldid=805938319 ''
Categories : Wikipedia articles incorporating text from the 20th edition of Gray 's Anatomy ( 1918 ) Shoulder Scapula Hidden categories : Articles containing
Greek - language text Medicine infobox template using GraySubject or GrayPage Medicine infobox template using Dorlands parameter Talk Contents About Wikipedia
Deutsch Español فارسی Français Galego Ido Italiano Magyar Português Svenska ไทย Українська Edit links This page was last edited on 18 October 2017 ,
</item 4>

<item 5>
Glenoid cavity
Glenoid cavity - wikipedia Glenoid cavity Jump to : navigation , search Glenoid cavity Costal surface of left scapula . Glenoid cavity shown in red . Glenoid
cavity shown in red . This cavity articulates with the head of the humerus . Details Identifiers Latin cavitas glenoidalis , fossa glenoidalis Dorlands /
Elsevier c_16 / 12220465 TA A02. 4.01. 019 FMA 23275 Anatomical terms of bone ( edit on Wikidata ) The glenoid cavity or glenoid fossa of scapula is a part of
the shoulder . It is a shallow , pyriform articular surface , which is located on the lateral angle of the scapula . It is directed laterally and forward and
articulates with the head of the humerus ; it is broader below than above and its vertical diameter is the longest . This cavity forms the glenohumeral joint
along with the humerus . This type of joint is classified as a synovial , ball and socket joint .
</item 5>

<item 6>
Glenoid cavity
Animation of glenohumeral joint . The muscle shown is supraspinatus muscle . See also ( edit ) Glenopolar angle Notes ( edit ) Jump up ^ The word glenoid is
pronounced / ˈɡliːnɔɪd / or / ˈɡlɛnɔɪd / ( both are common ) and is from Greek : gléne , `` socket '' , reflecting the shoulder joint 's ball - and - socket
form . References ( edit ) Jump up ^ `` Scapulohumeral Rhythm - Physiopedia , universal access to physiotherapy knowledge '' . www.physio-pedia.com . Retrieved
2016 - 09 - 01 . Jump up ^ Larson 2009 , p. 65 Jump up ^ Martin , A.J. ( 2006 ) . Introduction to the Study of Dinosaurs . Second Edition . Oxford , Blackwell
Publishing . pg. 299 - 300 . ISBN 1 - 4051 - 3413 - 5 .
</item 6>

<item 7>
Glenoid cavity
This type of joint is classified as a synovial , ball and socket joint . The humerus is held in place within the glenoid cavity by means of the long head of the
biceps tendon . This tendon originates on the superior margin of the glenoid cavity and loops over the shoulder , bracing humerus against the cavity . The
rotator cuff also reinforces this joint more specifically with the supraspinatus tendon to hold the head of the humerus in the glenoid cavity . The cavity
surface is covered with cartilage in the fresh state ; and its margins , slightly raised , give attachment to a fibrocartilaginous structure , the glenoid
labrum , which deepens the cavity . This cartilage is very susceptible to tearing . When torn , it is most commonly known as a SLAP lesion which is generally
caused by repetitive shoulder movements . Compared to the acetabulum ( at the hip - joint ) the glenoid cavity is relatively shallow . This makes the shoulder
joint prone to dislocation ( luxation ) . Strong glenohumeral ligaments and muscles prevents dislocation in most cases .
</item 7>

<item 8>
Glenoid cavity
Strong glenohumeral ligaments and muscles prevents dislocation in most cases . By being so shallow the glenoid cavity allows the shoulder joint to have the
greatest mobility of all joints in the body , allowing 120 degrees of unassisted flexion . Additional range of motion in shoulder flexion ( typically up to 180
degrees in humans ) is also accomplished by the great mobility of the scapula ( shoulder blade ) through a process known as scapulohumeral rhythm . Contents (
hide ) 1 Evolution 1.1 In dinosaurs 2 Additional images 3 See also 4 Notes 5 References 6 References 7 External links Evolution ( edit ) Interpretations of the
fossil remains of Australopithecus africanas ( STS 7 ) and A. afarensis ( AL 288 - 1 ; a.k.a. Lucy ) suggest that the glenoid fossa was oriented more cranially
in these species than in modern humans .
</item 8>

<item 9>
Glenoid cavity
at 16 : 19 . Text is available under the Creative Commons Attribution - ShareAlike License ; additional terms may apply . By using this site , you agree to the
Terms of Use and Privacy Policy . Wikipedia ® is a registered trademark of the Wikimedia Foundation , Inc. , a non-profit organization . About Wikipedia
</item 9>

<item 10>
Glenoid cavity
This reflects the importance of overhead limb postures and suggests a retention of arboreal adaptations in these hominoid primates , whereas the lateral
orientation of the glenoid in modern humans reflects the typical lowered position of the arm . In dinosaurs ( edit ) In dinosaurs the main bones of the pectoral
girdle were the scapula ( shoulder blade ) and the coracoid , both of which directly articulated with the clavicle . The place on the scapula where it
articulated with the humerus ( upper bone of the forelimb ) is called the glenoid . The glenoid is important because it defines the range of motion of the
humerus . Additional images ( edit ) Left scapula . Glenoid cavity shown in red . Animation . Glenoid cavity shown in red . Glenoid fossa of right side .
Anterior view of left scapula showing glenoid cavity `` 2 '' Animation of shoulder joint . The muscles shown are subscapularis muscle ( at right ) ,
infraspinatus muscle ( at top left ) , teres minor muscle ( at bottom left ) .
</item 10>
=== END ITEM id=dfdcf8a5a10acae1 ===

=== ITEM id=267aa0ab8d045eda ===
Gold sentences and facts:
Sentence 0: The soma ( cell body ) of each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root , where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons . The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
  Fact 0.0: The soma (cell body) of each pseudounipolar neuron is located within a dorsal root ganglion.
  Fact 0.1: The axon leaves the cell body.
  Fact 0.2: The axon leaves the dorsal root ganglion.
  Fact 0.3: The axon goes into the dorsal root.
  Fact 0.4: In the dorsal root, the axon splits into two branches.
  Fact 0.5: The central branch of the axon goes to the posterior (dorsal) horn of the spinal cord.
  Fact 0.6: The central branch forms synapses with other neurons.
  Fact 0.7: The peripheral branch travels through the distal dorsal root.
  Fact 0.8: The peripheral branch travels into the spinal nerve.
  Fact 0.9: The peripheral branch travels all the way until skin, joint, and muscle.

Retrieved items:
<item 1>
Pseudounipolar neuron
In the sensory ganglia of cranial nerves ( edit ) Pseudounipolar neurons are also found in the sensory ganglia of cranial nerves V , VII , IX and X , all mixed
- cranial nerves . The vestibulocochlear nerve , has the bipolar neuron type in both the spiral ganglion as well as Scarpa 's ganglion . Function ( edit )
Signal pathway ( edit ) These neurons have sensory receptors on skin , joints , muscles , and other parts of the body . The area of the axon that is closest to
the receptor is the trigger zone for the neuron . The signal is conducted through the axon to the dorsal root ganglion 's cell body , then through the axon and
ending at the sensory nuclei in the posterior horn of the spinal cord . See also ( edit ) Bipolar neuron Multipolar neuron Unipolar neuron References ( edit ) ^
Jump up to : Kandel , Eric R. ; Schwartz , James H. ; Mack , Thomas M. Jessell ( 2000 ) . Principles of neural Science ( 4th ed . ) . Cambridge : McGrawHill .
p. 24 .
</item 1>

<item 2>
Motor neuron
These columns are further explained in the table below . Motor Columns of Spinal Cord Motor Column Location in Spinal Cord Target Median Motor Column Present
entire length Axial muscles Hypaxial Motor Column Thoracic Region Body wall muscles Preganglionic Motor Column Thoracic Region Sympathetic ganglion Laterl Motor
Column Brachial and Lumbar region ( both regions are further divided into medial and lateral domains ) Muscles of the limbs Phrenic Motor Coulmn Cervical Region
Diaphragm Anatomy and Physiology ( edit ) Spinal cord tracts Location of lower motor neurons in spinal cord Upper motor neurons ( edit ) Upper motor neurons
originate in the motor cortex located in the precentral gyrus . The cells that make up the primary motor cortex are Betz cells , which are a type of pyramidal
cell . The axons of these cells descend from the cortex to form the corticospinal tract . White matter Tract ( edit ) White matter tracts are bundles of axons
and serve as means to propagate nerve impulses on a large scale . These are found in the spinal cord and appear white from the myelin surrounding axons .
</item 2>

<item 3>
Motor neuron
the command of visceral muscles is disynaptic ( involving two neurons : the general visceral motor neuron located in the CNS , which synapses onto a ganglionic
neuron , located in the PNS , which synapses onto the muscle ) . All vertebrate motor neurons are cholinergic , that is , they release the neurotransmitter
acetylcholine . Parasympathetic ganglionic neurons are also cholinergic , whereas most sympathetic ganglionic neurons are noradrenergic , that is , they release
the neurotransmitter noradrenaline . ( see Table ) Motor neurons in Regenerative medicine ( edit ) Human lower motor neurons can be generated in vitro from
embryonic stem cells and induced pluripotent stem cells . See also ( edit ) Betz cell Central chromatolysis Motor dysfunction Nerve References ( edit ) Jump up
^ Tortora , Gerard ; Derrickson , Bryan ( 2014 ) . Principles of Anatomy & Physiology ( 14th ed . ) . New Jersey : John Wiley & Sons , Inc . pp. 406 , 502 , 541
. ISBN 978 - 1 - 118 - 34500 - 9 .
</item 3>

<item 4>
Motor neuron
Nature Neuroscience. 15 ( 12 ) : 1636 -- 1644 . doi : 10.1038 / nn. 3242 . ISSN 1097 - 6256 . PMC 3676175 . PMID 23103965 . Jump up ^ Fitzpatrick , D. ( 2001 )
The Primary Motor Cortex : Upper Motor Neurons That Initiate Complex Voluntary Movements . In D. Purves , G.J. Augustine , D. Fitzpatrick , et al. ( Ed . ) ,
Neuroscience . Retrieved from https://www.ncbi.nlm.nih.gov/books/NBK10962/ Jump up ^ Tortora , G.J. , Derrickson , B. ( 2011 ) . The Spinal Cord and Spinal
Nerves . In B. Roesch , L. Elfers , K. Trost , et al. ( Ed . ) , Principles of Anatomy and Physiology ( pp. 443 - 468 ) . New Jersey : John Wiley & Sons , Inc .
Jump up ^ Fitzpatrick , D.
</item 4>

<item 5>
Pseudounipolar neuron
Pseudounipolar neuron - wikipedia Pseudounipolar neuron Jump to : navigation , search Pseudounipolar neuron 1 : Pseudounipolar neuron , 2 : Bipolar neuron
Details System Nervous system Identifiers Latin neuron pseudounipolare TH H2. 00.06. 1.00047 Anatomical terms of neuroanatomy ( edit on Wikidata ) A
pseudounipolar neuron ( pseudo -- false , uni -- one ) is a kind of sensory neuron in the peripheral nervous system . This neuron contains an axon that has
split into two branches ; one branch runs to the periphery and the other to the spinal cord . Contents ( hide ) 1 Structure 1.1 In the dorsal root ganglia 1.2
In the sensory ganglia of cranial nerves 2 Function 2.1 Signal pathway 3 See also 4 References Structure ( edit ) By definition , a pseudounipolar neuron has
one axon with two branches : central and peripheral . These axonal branches should not be confused with dendrites .
</item 5>

<item 6>
Pseudounipolar neuron
Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers Postganglionic nerve fibers ) Connective tissues
Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin incisure Node of Ranvier Internodal segment Satellite
glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon Telodendron Axon terminals Axoplasm Axolemma Neurofibril / neurofilament Dendrite Nissl body
Dendritic spine Apical dendrite / Basal dendrite Types Bipolar Unipolar Pseudounipolar Multipolar Interneuron Renshaw Afferent nerve fiber / Sensory neuron GSA
GVA SSA SVA fibers Ia or Aα Ib or Golgi or Aα II or Aβ and Aγ III or Aδ or fast pain IV or C or slow pain Efferent nerve fiber / Motor neuron GSE GVE SVE Upper
motor neuron Lower motor neuron α motorneuron β motorneuron γ motorneuron Termination Synapse Electrical synapse / Gap junction Chemical
</item 6>

<item 7>
Motor neuron
; Mallatt , J. ( 1997 ) . Human Anatomy ( 2nd ed . ) . Menlo Park , CA : Benjamin / Cummings . ISBN 0 - 8053 - 4068 - 8 . ( hide ) Nervous tissue CNS Tissue
Types Grey matter White matter Projection fibers Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil
Meninges Cell Types Neuronal Pyramidal Purkinje Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells
Tanycyte Microglia PNS General Dorsal Root Ganglion Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion ( Preganglionic nerve fibers
Postganglionic nerve fibers ) Connective tissues Epineurium Perineurium Endoneurium Nerve fascicle Neuroglia Myelination : Schwann cell Neurilemma Myelin
incisure Node of Ranvier Internodal segment Satellite glial cell Neurons / nerve fibers Parts Soma Axon hillock Axon
</item 7>

<item 8>
Pseudounipolar neuron
These axonal branches should not be confused with dendrites . These sensory neurons are an exception to the typical neuron , in that they do not have separate
dendrites and an axonal process , but rather one branched process that serves both functions . The axon has a peripheral branch ( from the cell body to the
periphery : skin , joint and muscle ) and a central branch ( from the cell body to spinal cord ) . In the dorsal root ganglia ( edit ) The soma ( cell body ) of
each pseudounipolar neuron is located within a dorsal root ganglion . The axon leaves the cell body ( and out of the dorsal root ganglion ) into the dorsal root
, where it splits into two branches . The central branch goes to the posterior ( dorsal ) horn of the spinal cord , where it forms synapses with other neurons .
The peripheral branch travels through the distal dorsal root into the spinal nerve all the way until skin , joint , and muscle .
</item 8>

<item 9>
Pseudounipolar neuron
. Cambridge : McGrawHill . p. 24 . ISBN 978 - 0838577011 . ^ Jump up to : MacCallum , Don . `` Peripheral Nervous System '' . Histology and Virtual Microscopy
Learning Resources . University of Michigan Medical School . Archived from the original on 7 July 2014 . Retrieved 24 June 2014 . Jump up ^ Miller 's Anatomy of
the Dog , By Howard E. Evans , Alexander de Lahunta , 4th ed , pp 565 ( hide ) Nervous tissue CNS Tissue Types Grey matter White matter Projection fibers
Association fiber Commissural fiber Lemniscus Funiculus Fasciculus Nerve tract Decussation Commissure Neuropil Meninges Cell Types Neuronal Pyramidal Purkinje
Granule Glial insulating : Myelination : Oligodendrocyte other Astrocyte Radial glial cell Ependymal cells Tanycyte Microglia PNS General Dorsal Root Ganglion
Ramus Ventral Root Ramus Ramus communicans Gray White Autonomic ganglion (
</item 9>

<item 10>
Motor neuron
These are found in the spinal cord and appear white from the myelin surrounding axons . While these tracts exist for both efferent and afferent nerves ,
efferent nerves will carry signals down the spinal cord toward their effector target . These tracts also serve as the place of origin for lower motor neurons .
There are seven major white matter motor tracts to be found in the spinal cord and are listed below : Lateral Corticospinal Tract Rubrospinal Tract Lateral
Reticulospinal Tract Vestibulospinal tract Medial Reticulospinal Tract Tectospinal Tract Anterior Corticospinal Tract Lower motor neurons ( edit ) Lower motor
neurons are those that originate in the spinal cord and directly or indirectly innervate effector targets . The target of these neurons varies , but in the
somatic nervous system the target will be some sort of muscle fiber . There are three primary categories lower motor neurons , which can be further divided in
sub-categories .
</item 10>
=== END ITEM id=267aa0ab8d045eda ===
