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

This file has 10 items.

=== ITEM id=6d3f7a8d3ebf8bc6 ===
Gold sentences and facts:
Sentence 0: Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant", his ten-novel fantasy series.
  Fact 0.0: Stephen Reeder Donaldson was born on May 13, 1947.
  Fact 0.1: Stephen Reeder Donaldson is an American novelist.
  Fact 0.2: Stephen Reeder Donaldson writes fantasy novels.
  Fact 0.3: Stephen Reeder Donaldson writes science fiction novels.
  Fact 0.4: Stephen Reeder Donaldson writes mystery novels.
  Fact 0.5: Stephen Reeder Donaldson is most famous for "The Chronicles of Thomas Covenant".
  Fact 0.6: "The Chronicles of Thomas Covenant" is a ten-novel fantasy series.
  Fact 0.7: "The Chronicles of Thomas Covenant" is a series by Stephen Reeder Donaldson.
Sentence 1: Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.
  Fact 1.0: Michael John Moorcock was born on 18 December 1939.
  Fact 1.1: Michael John Moorcock is an English writer.
  Fact 1.2: Michael John Moorcock is primarily a writer of science fiction.
  Fact 1.3: Michael John Moorcock is primarily a writer of fantasy.
  Fact 1.4: Michael John Moorcock has also published literary novels.

Retrieved items:
<item 1>
Orson Scott Card
Orson Scott Card (born August 24, 1951) is an American novelist, critic, public speaker, essayist, and columnist.

 He writes in several genres but is known best for science fiction.

 His novel "Ender's Game" (1985) and its sequel "Speaker for the Dead" (1986) both won Hugo and Nebula Awards, making Card the only author to win both science
fiction's top U.S. prizes in consecutive years.

 A feature film adaptation of "Ender's Game", which Card co-produced, was released in late October 2013 in Europe and on November 1, 2013, in North America.
</item 1>

<item 2>
The Chinese Agent
The Chinese Agent (1970) is a comic novel by Michael Moorcock.

 It is a revision of "Somewhere in the Night", which Moorcock published in 1966 under the pseudonym Bill Barclay.

 Although Moorcock is best known as the author of fantasy fiction and science fiction-based parables such as "Behold the Man" and "The Dancers at the End of
Time", here he writes a light-hearted caper that parodies the spy novel genre.
</item 2>

<item 3>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 3>

<item 4>
Douglas Hill
Douglas Arthur Hill (6 April 1935 – 21 June 2007) was a Canadian science fiction author, editor and reviewer.

 He was born in Brandon, Manitoba, the son of a railroad engineer, and was raised in Prince Albert, Saskatchewan.

 An avid science fiction reader from an early age, he studied English at the University of Saskatchewan (where he earned an Honours B.A. in 1957) and at the
University of Toronto.

 He married fellow writer and U. of S. alumna Gail Robinson (writer) in 1958; they moved to Britain in 1959, where he worked as a freelance writer and editor
for Aldus Books.

 In 1967–1968 he served as Assistant Editor of the controversial "New Worlds" science fiction magazine under Michael Moorcock.
</item 4>

<item 5>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 5>

<item 6>
The Final Programme
The Final Programme is a novel by British science fiction and fantasy writer Michael Moorcock.

 Written in 1965 as the underground culture was beginning to emerge, it was not published for several years.

 Moorcock has stated that publishers at the time considered it was "too freaky".
</item 6>

<item 7>
Warlord of the Air
The Warlord of the Air is a 1971 British alternate history science fiction novel written by Michael Moorcock.

 It concerns the adventures of Oswald Bastable, an Edwardian era soldier stationed in India, and his adventures in an alternate universe, in his own future,
wherein the First World War never happened.

 It is the first part of Moorcock's 'A Nomad of the Time Streams' trilogy and, in its use of speculative technology (such as airships) juxtaposed against an
Edwardian setting, it is widely considered to be one of the first steampunk novels.

 The novel was first published by Ace Books as part of their Ace Science Fiction Specials series.
</item 7>

<item 8>
Michael Moorcock
Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.

 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
</item 8>

<item 9>
A Cure for Cancer
A Cure for Cancer is a novel by British fantasy and science fiction writer Michael Moorcock, first published in London 1971 by Allison and Busby.

 The book is part of Moorcock's long-running Jerry Cornelius series.
</item 9>

<item 10>
Behold the Man (novel)
Behold the Man (1969) is a science fiction novel by British writer Michael Moorcock.

 It originally appeared as a novella in a 1966 issue of "New Worlds"; later, Moorcock produced an expanded version which was first published in 1969 by Allison
& Busby.

 The title derives from the Gospel of John, Chapter 19, Verse 5: "Then Jesus came out, wearing the crown of thorns and the purple robe.

 And Pilate said to them Behold the Man."
</item 10>
=== END ITEM id=6d3f7a8d3ebf8bc6 ===

=== ITEM id=dad061991040b89d ===
Gold sentences and facts:
Sentence 0: And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
  Fact 0.0: And There's More is a comedy sketch show.
  Fact 0.1: And There's More stars Jimmy Cricket.
  Fact 0.2: And There's More was produced by Central Independent Television.
  Fact 0.3: And There's More was produced for ITV.
  Fact 0.4: And There's More was produced from 1985 till 1988.
Sentence 1:  He currently lives with his family in Rochdale, North West England.
  Fact 1.0: The person described in this passage currently lives with his family in Rochdale, North West England.
  Fact 1.1: The person described in this passage currently lives in Rochdale.
  Fact 1.2: Rochdale is in North West England.

Retrieved items:
<item 1>
Standard Star Building
The Standard Star Building is a historic commercial structure located in the Downtown section of New Rochelle, Westchester County, New York.

 The building, designed by architect Lawrence J. Barnard, was completed in 1924 and is an architecturally significant example of the Italian Renaissance style
in New Rochelle.

 Although it has been altered, these changes occurred only on the Le Count Place façade and the interior of the building.

 The original Standard Star building exterior remains largely unchanged.

 It is further historically significant for its association with a long-published New Rochelle newspaper covering life in New Rochelle.
</item 1>

<item 2>
Hotel Kenmore Hall
Hotel Kenmore Hall is a 22-story single room occupancy hotel located at 145 East 23rd Street in the Gramercy section of Manhattan, designed by architect Maurice
Deutsch and constructed in 1927.

 Author Nathanael West lived and worked at the hotel as a night manager in the early years after the hotel opened; one of West's real-life experiences at the
hotel inspired the incident between Romola Martin and Homer Simpson that would later appear in "The Day of the Locust" (1939).

 West allowed friends like Edmund Wilson, Erskine Caldwell, S. J. Perelman and Maxwell Bodenheim free room and meals.

 Dashiell Hammett finished "The Maltese Falcon" here
</item 2>

<item 3>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 3>

<item 4>
Terry Ray (actor)
He wrote and starred in two more shows at the Dorothy Chandler Pavilion, one co-starring Amanda Bynes and the other Hilary Duff.

<br><br>Terry is also the voice of the "Gaydar Gun", a spinoff toy from the film Gaydar.
</item 4>

<item 5>
Rubin Schron
Other buildings he owns include the Monterey, a 521-unit rental multifamily building on Manhattan's Upper East Side; over the decades, Schron has also amassed a
portfolio of Mitchell-Lama apartment buildings whose values have been skyrocketing to record values, after reverting to market rates when government subsidies
expired.

 In 2007, he sold nearly 4,000 units of former Mitchell-Lama properties in five complexes in Harlem and on Roosevelt Island for $940 million.

 Schron, who practices Orthodox Judaism, has eight children and 50 grandchildren.

 He and his family have lived in the same single-family home in Brooklyn for many decades.
</item 5>

<item 6>
John Garabedian
John H. Garabedian is an American radio personality and disc jockey born on December 20, 1941 (age 75).

 He is best known as the creator and former long-time host of "Open House Party".

 He has been involved in Massachusetts radio and television stations for over fifty years.

 Garabedian currently owns three homes, all in New England: one in Southborough, Massachusetts, another on Cape Cod, and a cabin-like home in Cabot, Vermont.
</item 6>

<item 7>
Terry Ray (actor)
Terry Ray (born February 12, 1961) is an American actor, screenwriter, and producer.

 Some of his work includes "Gaydar" and "Cost of Living" (2009).

 Ray is the creator and writer of "here!

 TV"'s sitcom "From Here on OUT", the first original gay sitcom created by a gay network.

 Terry stars in the sitcom, along with Juliet Mills, Suzanne Whang and T.J. Hoban.

<br><br>Terry Ray's 1987 appearance on Scrabble (game show) is noted as number 3 in the "10 Great Gay Moments in Game Show History", after Charles Nelson Reilly
and Paul Lynde.

 In the same appearance, Terry was also named as one of the "5 Weirdest Game Show Contestants Ever", by Movieline.

<br><br>Terry’s first professional writing job was the stage performance “Hollywood Goes Classical" at the venerable Dorothy Chandler Pavilion starring Mickey
Rooney, Michael York, Dean Jones, Rhonda Fleming and several other classic film stars.
</item 7>

<item 8>
And There's More
And There's More is a comedy sketch show starring Jimmy Cricket produced by Central Independent Television for ITV from 1985 till 1988.
</item 8>

<item 9>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 9>

<item 10>
Richard Pleasance
Richard Arnold Pleasance is an Australian rock musician and producer.

 He was a founding member of Boom Crash Opera on guitar, bass guitar, vocals and as a songwriter in 1985; they released three albums before Pleasance left in
1992.

 Their hit Australian Recording Industry Association (ARIA) singles, "Great Wall" (No. 5, 1986) and "Onion Skin" (No. 11, 1989) were co-written by Pleasance,
who also co-produced their second album, "These Here Are Crazy Times" (No. 10, 1989).

 His debut solo release, "Galleon" received four nominations at the ARIA Music Awards for 1992.

 Pleasance composed the theme music for Australian television series, "SeaChange" (1998–2001), for 2006 feature film "Kenny" and more recently he composed the
theme music for the prison drama series "Wentworth".

 Pleasance is married to Michelle and, as from May 2009, he was living in Hepburn Springs, Victoria where he has a recording studio.
</item 10>
=== END ITEM id=dad061991040b89d ===

=== ITEM id=afa626ea20e4eb03 ===
Gold sentences and facts:
Sentence 0:  Levitt & Sons developed the Monmouth Heights section from 1965-1969.
  Fact 0.0: Levitt & Sons developed the Monmouth Heights section from 1965 to 1969.
Sentence 1:  The company is most famous for having built the town of Levittown, New York.
  Fact 1.0: Levitt & Sons is most famous for having built the town of Levittown, New York.

Retrieved items:
<item 1>
Newark Bay
Newark Bay is a tidal bay at the confluence of the Passaic and Hackensack Rivers in northeastern New Jersey.

 It is home to the Port Newark-Elizabeth Marine Terminal, the largest container shipping facility in Port of New York and New Jersey, the third largest and one
of the busiest in the United States.

 An estuary, it is periodically dredged to accommodate ocean-going ships.
</item 1>

<item 2>
Province of New Jersey
The Province of New Jersey was one of the Middle Colonies of Colonial America and became the U.S. state of New Jersey in 1776.

 The province had originally been settled by Europeans as part of New Netherland, but came under English rule after the surrender of Fort Amsterdam in 1664,
becoming a proprietary colony.

 The English then renamed the province after the Isle of Jersey in the English Channel.

 The Dutch Republic reasserted control for a brief period in 1673–1674.

 After that it consisted of two political divisions, East Jersey and West Jersey, until they were united as a royal colony in 1702.

 The original boundaries of the province were slightly larger than the current state, extending into a part of the present state of New York, until the border
was finalized in 1773.
</item 2>

<item 3>
HMS Monmouth
Seven ships of the Royal Navy have been named HMS "Monmouth".

 Monmouth was the name of a castle and is now the name of a town in Wales; the name also recognises James Scott, 1st Duke of Monmouth, the "Black Duke".

 The first HMS "Monmouth" was named after the town but whilst the second one was being used the Monmouth Rebellion took place and this caused some concern to
the crew.

 Since that time, all of "Monmouth"s have had their names painted in black and the name and the crest are covered because of the shame of the Duke of Monmouth's
execution.

 Today the officers of the latest HMS "Monmouth" carry black handkerchiefs and the ship is known as the Black Duke.

 However the town of Monmouth still gives the freedom of the town to the ship's crew and talks of the "enduring link" between the town and the ship.
</item 3>

<item 4>
Riverton, New York
Riverton, New York was a planned community built in 1973 in the town of Henrietta, New York, along the Genesee River.

 Riverton was seventh of the thirteen communities receiving Federal guarantee assistance from HUD as part of their New Communities Program created by the Urban
Growth and New Community Development Act of 1970.

 $12 million was earmarked for the community.

 Riverton was supposed to encompass a 2335 acre land area bordered on the north by the New York State Thruway, on the west by the Genesee River, on the east by
East River Rd., and on the south by the Rush-Henrietta town line, with some development continuing east along Erie Station Rd.

 Original plans called for construction to occur for a 16-year period and eventually house over 25,000 people, with the community including an 18-hole golf
course, 12 swimming pools, 3 artificial lakes, a marina on the Genesee, and commercial and industrial centers.

 However, only a small portion of this was built along Scottsville-West Henrietta Rd.
</item 4>

<item 5>
Oakland Mills, New Jersey
Oakland Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The area is named for Oakland Mill, a mill along the Manalapan Brook operated by the Boo family.

 Some areas within the Manalapan Brook valley and surround forestland are preserved while the remainder of the area surrounding Oakland Mills has since been
developed into housing developments.
</item 5>

<item 6>
Hanbys Corner, Delaware
Hanbys Corner is an unincorporated community in New Castle County, Delaware, United States.

 It is located at the intersection of Delaware Route 3 (Marsh Road) and Delaware Route 92 (Naamans Road), in Brandywine Hundred.

 The area is named for Richard G. Hanby, who first purchased the 125 acre parcel from the descendants of William Penn in 1753.

 His descendants included several notable figures in the local political scene including Samuel Winfield Hanby (1817-1892) who was elected as a State
Representative in 1874, Jacob Klose Hanby (1839-1932) who was Samuel's son and was elected State Representative in 1904, Robert Johnson Hanby (1834-1898), who
served in the 124th PA infantry during the Civil War and was elected State Senator in 1896, and Florence Wood Hanby (1870-1963), who was Robert's
daughter-in-law and was the first woman elected to the Delaware House of Representatives in 1924.

 In addition Wayne Hanby and James Hanby have both served as Justices of the Peace for New Castle County.
</item 6>

<item 7>
Monmouth Heights at Marlboro, New Jersey
Monmouth Heights at Marlboro is an unincorporated community located within Marlboro Township in Monmouth County, New Jersey, United States.

 The housing development was created in the 1960s by Levitt & Sons though some protests occurred by long-time residents of Marlboro Township at the time of its
creation.

 The community is located along New Jersey Route 79 across the highway from Marlboro High School.
</item 7>

<item 8>
Taylors Mills, New Jersey
Taylors Mills is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 The main road that runs through the area is named for the community, Taylors Mills Road.
</item 8>

<item 9>
Riverton, New York
However, only a small portion of this was built along Scottsville-West Henrietta Rd.

 Created as part of the community were Riverton Knolls, a moderately priced townhouse subdivision, the Riverton Golf Club, a small 9-hole golf course, and the
Riverton Park, which included a playground, baseball diamond, river front benches, walking trail, and a volleyball court.

 Also nestled in a bend in the Genesee was a 28 acre forested nature park.
</item 9>

<item 10>
Monmouth Heights at Manalapan, New Jersey
Monmouth Heights at Manalapan is an unincorporated community located within Manalapan Township in Monmouth County, New Jersey, United States.

 Levitt & Sons developed the Monmouth Heights section from 1965-1969.

 Residents of the 685 homes in the development are required to join the Monmouth Heights Community Association, which provides pools and other amenities.
</item 10>
=== END ITEM id=afa626ea20e4eb03 ===

=== ITEM id=34492ee2d29a1d18 ===
Gold sentences and facts:
Sentence 0: Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.
  Fact 0.0: Bluebeard's Castle is a one-act opera.
  Fact 0.1: Bluebeard's Castle is by Hungarian composer Béla Bartók.
  Fact 0.2: The Hungarian title of Bluebeard's Castle is A kékszakállú herceg vára.
  Fact 0.3: Literally, the Hungarian title means The Blue-Bearded Duke's Castle.
Sentence 1: Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.
  Fact 1.0: Agrippina (HWV 6) is an opera seria.
  Fact 1.1: Agrippina (HWV 6) has three acts.
  Fact 1.2: Agrippina (HWV 6) is by George Frideric Handel.
  Fact 1.3: Agrippina (HWV 6) has a libretto by Cardinal Vincenzo Grimani.

Retrieved items:
<item 1>
The Padlock
The Padlock is a two-act 'afterpiece' opera by Charles Dibdin.

 The text was by Isaac Bickerstaffe.

 It debuted in 1768 at the Drury Lane Theatre in London as a companion piece to "The Earl of Warwick".

 It partnered other plays before a run of six performances in tandem with "The Fatal Discovery" by John Home.

 "The Padlock" was a success, largely due to Dibdin's portrayal of Mungo, a blackface caricature of a black servant from the West Indies.

 The company took the production to the United States the next year, where a portrayal by Lewis Hallam, Jr. as Mungo met with even greater accolades.

 The libretto was first published in London in 1768 (?)

 and in Dublin in 1775.

 The play remained in regular circulation in the U.S. as late as 1843.

 It was revived by the Old Vic Company in London and on tour in the UK in 1979 in a new orchestration by Don Fraser and played in a double-bill with Garrick's
"Miss In Her Teens".
</item 1>

<item 2>
Orfeo ed Euridice
Orfeo ed Euridice (French version: Orphée et Eurydice ; English: "Orpheus and Eurydice") is an opera composed by Christoph Willibald Gluck based on the myth of
Orpheus, set to a libretto by Ranieri de' Calzabigi.

 It belongs to the genre of the "azione teatrale", meaning an opera on a mythological subject with choruses and dancing.

 The piece was first performed at the Burgtheater in Vienna on 5 October 1762 in the presence of Empress Maria Theresa.

 "Orfeo ed Euridice" is the first of Gluck's "reform" operas, in which he attempted to replace the abstruse plots and overly complex music of "opera seria" with
a "noble simplicity" in both the music and the drama.
</item 2>

<item 3>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 3>

<item 4>
Armand Castelmary
Armand Castelmary, real name Comte Armand de Castan, born Toulouse 16 August 1834, died New York City 10 February 1897, was a French operatic bass.

 He created roles in three major premieres at the Paris Opera – Don Diego in "L'Africaine" by Meyerbeer (1865), the Monk in Verdi's "Don Carlos" (1867), and
Horatio in Ambroise Thomas's "Hamlet" (1868).

 Castelmary also appeared at opera houses in England and the United States, and died onstage at the Metropolitan Opera House, New York, during a performance of
"Martha" by Friedrich von Flotow.
</item 4>

<item 5>
The Carp (opera)
The Carp is a one-act comic opera (styled "a whimsicality") with a libretto by Frank Desprez and music by Alfred Cellier.

 It was first produced at the Savoy Theatre from 13 February 1886 to 19 January 1887, as a companion piece to "The Mikado".

 It was then revived as companion to "Ruddigore" from 21 February 1887 to 5 November 1887.

 The piece also toured throughout 1888.
</item 5>

<item 6>
Daphne (opera)
Daphne, Op.

82, is an opera in one act by Richard Strauss, subtitled "Bucolic Tragedy in One Act".

 The German libretto was by Joseph Gregor.

 The opera is based loosely on the mythological figure Daphne from Ovid's "Metamorphoses" and includes elements taken from "The Bacchae" by Euripides.

 The opera premiered at the Semperoper in Dresden on 15 October 1938, originally intended as a double bill with Strauss' "Friedenstag", but as the scale of
"Daphne" grew, that idea was abandoned.

 The conductor of the first performance was Karl Böhm, to whom the opera was dedicated.
</item 6>

<item 7>
Agrippina (opera)
Agrippina (HWV 6) is an "opera seria" in three acts by George Frideric Handel with a libretto by Cardinal Vincenzo Grimani.

 Composed for the 1709–10 Venice "Carnevale" season, the opera tells the story of Agrippina, the mother of Nero, as she plots the downfall of the Roman Emperor
Claudius and the installation of her son as emperor.

 Grimani's libretto, considered one of the best that Handel set, is an "anti-heroic satirical comedy", full of topical political allusions.

 Some analysts believe that it reflects Grimani's political and diplomatic rivalry with Pope Clement XI.
</item 7>

<item 8>
Bernhard Bötel
His stage repertoire included Belmonte in Wolfgang Amadeus Mozart's "Die Entführung aus dem Serail", Chapelou in Adolphe Adam's "Le postillon de Lonjumeau",
Count Almaviva in Gioachino Rossini's "The Barber of Seville", Daniel in Franz Lehár's "The Merry Widow", the Duke of Mantua in Giuseppe Verdi's "Rigoletto",
Gabriel von Eisenstein in Johann Strauss II's "Die Fledermaus", Indigo in Strauss' "Indigo und die vierzig Räuber", Jeník in Bedřich Smetana's "The Bartered
Bride", Paolino in Domenico Cimarosa's "Il matrimonio segreto", Pâris in Jacques Offenbach's "La belle Hélène", Pietro in Franz von Suppé's "Boccaccio", and
Wilhelm Meister in Ambroise Thomas' "Mignon".
</item 8>

<item 9>
Senza sangue
Senza sangue (Without blood) an opera in one act by Péter Eötvös with a libretto by Mari Mezei.

 It is based on the second part of the novel of the same name by Alessandro Baricco and is intended to be performed jointly with Béla Bartók's one-act
"Bluebeard's Castle".

 The concert première was given by the Cologne Philharmonic Orchestra on 1 May 2015.

 It was premièred in full on 15 May 2016 at the Festival d'Avignon.

 It is his tenth opera.

 Anne Sofie von Otter, who gave the New York première with the New York Philharmonic, said: "“The piece is not at all easy for the two singers... [It’s] hard to
find the pitch; you have to work it into the voice, unless of course you happen to have perfect pitch, which I don’t."
</item 9>

<item 10>
The Heiress (1786 play)
The Heiress is a comedy play by the British playwright and soldier John Burgoyne.

 The play debuted at the Drury Lane Theatre on 14 January 1786.

 It concerns the engagement of Lord Gayville to Miss Alscrip, a fashionable woman he believes to be an heiress.

 Gayville later discovers that the woman who really stands to inherit the fortune is his true love Miss Clifford.

 The play was an enormous success, running for 31 performances in its initial season and being revived again the following year.

 The play was initially anonymous, but Burgoyne was soon widely reported to be the author and he acknowledged this after the play's debut.
</item 10>
=== END ITEM id=34492ee2d29a1d18 ===

=== ITEM id=e777bb0057f44e55 ===
Gold sentences and facts:
Sentence 0: Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television host/personality who "TV Guide" called "The Charlie Chaplin of Television".
  Fact 0.0: Alan Young was born Angus Young.
  Fact 0.1: Alan Young was born on November 19, 1919.
  Fact 0.2: Alan Young died on May 19, 2016.
  Fact 0.3: Alan Young was a British-born Canadian-American actor.
  Fact 0.4: Alan Young was a voice actor.
  Fact 0.5: Alan Young was a comedian.
  Fact 0.6: Alan Young was a radio and television host/personality.
  Fact 0.7: TV Guide called Alan Young "The Charlie Chaplin of Television".
Sentence 1:  Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983) and in various other films, TV series and video games until his death.
  Fact 1.0: Alan Young was the voice of Disney's Scrooge McDuck for over thirty years.
  Fact 1.1: Alan Young first voiced Scrooge McDuck in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983).
  Fact 1.2: Alan Young voiced Scrooge McDuck in various other films, TV series and video games.
  Fact 1.3: Alan Young voiced Scrooge McDuck until his death.
Sentence 2:  During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy Awards in 1951.
  Fact 2.0: During the 1940s and 1950s, Alan Young starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television.
  Fact 2.1: The television version of "The Alan Young Show" gained Alan Young two Emmy Awards in 1951.
Sentence 3:  The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.
  Fact 3.0: The cartoon is an adaptation of Charles Dickens' "A Christmas Carol".
  Fact 3.1: The cartoon stars Scrooge McDuck as Ebenezer Scrooge.

Retrieved items:
<item 1>
Scrooge (1935 film)
Scrooge is a 1935 British fantasy film directed by Henry Edwards and starring Seymour Hicks, Donald Calthrop and Robert Cochran.

 Hicks appears as Ebenezer Scrooge, the miser who hates Christmas.

 It was the first sound version of the Charles Dickens classic "A Christmas Carol", not counting a 1928 short subject that now appears to be lost.

 Hicks had previously played the role of Scrooge on the stage many times beginning in 1901, and again in a 1913 British silent film version.
</item 1>

<item 2>
Alan Young
Alan Young (born Angus Young; November 19, 1919 – May 19, 2016) was a British-born Canadian-American actor, voice actor, comedian and radio and television
host/personality who "TV Guide" called "The Charlie Chaplin of Television".

 He was best known for his role as naive Wilbur Post in the television comedy series "Mister Ed" (1961–1966).

 Young was also the voice of Disney's Scrooge McDuck for over thirty years, first in the Academy Award-nominated short film "Mickey's Christmas Carol" (1983)
and in various other films, TV series and video games until his death.

 During the 1940s and 1950s, he starred in his own variety/comedy sketch shows "The Alan Young Show" on radio and television, the latter gaining him two Emmy
Awards in 1951.

 He also appeared in a number of feature films, starting from 1946, including the 1960 film "The Time Machine" and from the 1980s gaining a new generation of
viewers appearing in numerous Walt Disney Productions films as both an actor and voice actor.
</item 2>

<item 3>
Mickey's Christmas Carol
Mickey's Christmas Carol is a 1983 American animated featurette produced by Walt Disney Productions and released by Buena Vista Distribution.

 It was directed and produced by Burny Mattinson.

 The cartoon is an adaptation of Charles Dickens' "A Christmas Carol", starring Scrooge McDuck as Ebenezer Scrooge.

 Many other Disney characters, primarily from the Mickey Mouse universe, "Robin Hood", and "The Adventures of Ichabod and Mr. Toad", were cast throughout the
film.
</item 3>

<item 4>
Scrooge, or, Marley's Ghost
Scrooge, or, Marley's Ghost is a 1901 British short silent drama film, directed by Walter R. Booth, featuring the miserly Ebenezer Scrooge confronted by
Marley's ghost and given visions of Christmas past, present, and future, is the earliest known film adaptation of Charles Dickens's 1843 novel "A Christmas
Carol".

 The film, "although somewhat flat and stage-bound to modern eyes," according to Ewan Davidson of BFI Screenonline, "was an ambitious undertaking at the time,"
as, "not only did it attempt to tell an 80 page story in five minutes, but it featured impressive trick effects, superimposing Marley's face over the door
knocker and the scenes from his youth over a black curtain in Scrooge's bedroom."
</item 4>

<item 5>
A Christmas Carol
A Christmas Carol in Prose, Being a Ghost-Story of Christmas, commonly known as A Christmas Carol, is a novella by Charles Dickens, first published in London by
Chapman & Hall in 1843; the first edition was illustrated by John Leech.

 "A Christmas Carol" tells the story of Ebenezer Scrooge, an old miser who is visited by the ghost of his former business partner Jacob Marley and the Ghosts of
Christmas Past, Present and Yet to Come.

 After their visits Scrooge is transformed into a kinder, gentler man.
</item 5>

<item 6>
Scrooge (1913 film)
Scrooge is a 1913 British black and white silent film based on the 1843 novel "A Christmas Carol" by Charles Dickens.

 It starred Seymour Hicks as Ebenezer Scrooge.

 In the United States it was released in 1926 as "Old Scrooge".
</item 6>

<item 7>
Ebenezer (film)
Ebenezer is a 1998 Canadian made-for-television re-telling of Charles Dickens' classic "A Christmas Carol" with Jack Palance giving a performance as Ebenezer
Scrooge, á la Western genre.

 After a half-century of screen presence, in one of Palance's final projects before his retirement.
</item 7>

<item 8>
Scar (Disney)
Scar is a fictional character who appears in Walt Disney Pictures' 32nd animated feature film "The Lion King" (1994).

 The character is voiced by English actor Jeremy Irons, while his singing voice is provided by both Irons and American actor Jim Cummings, the latter of whom
was hired to replace Irons when the former damaged his singing voice.

 Subsequently, Scar makes minor appearances in the film's sequel " "(1998) and "The Lion King 1½ "(2004), in both of which he is voiced entirely by Cummings, as
well as appearing in the Broadway musical adaptation of the film, in which the role of Scar was originated by American actor John Vickery.
</item 8>

<item 9>
A Christmas Carol (2009 film)
A Christmas Carol is a 2009 American 3D computer animated motion-capture fantasy film written and directed by Robert Zemeckis.

 It is an adaptation of the Charles Dickens story of the same name and stars Jim Carrey in a multitude of roles, including Ebenezer Scrooge as a young,
middle-aged, and old man, and the three ghosts who haunt Scrooge.

 The film also features supporting roles done by Gary Oldman, Colin Firth, Bob Hoskins, Robin Wright, and Cary Elwes.
</item 9>

<item 10>
Scrooge (1951 film)
Scrooge is a 1951 British fantasy drama film and an adaptation of Charles Dickens' "A Christmas Carol" (1843).

 It stars Alastair Sim as Ebenezer Scrooge, and was produced and directed by Brian Desmond Hurst, with a screenplay by Noel Langley.

 It was released as A Christmas Carol in the United States.
</item 10>
=== END ITEM id=e777bb0057f44e55 ===

=== ITEM id=73da6115c776c6d0 ===
Gold sentences and facts:
Sentence 0:  It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style theme.
  Fact 0.0: The referenced piece plays during the opening sequence.
  Fact 0.1: Danny Elfman composed the piece in 1989.
  Fact 0.2: Danny Elfman composed the piece after Matt Groening approached him.
  Fact 0.3: Matt Groening is the series creator.
  Fact 0.4: Matt Groening requested a retro-style theme.
Sentence 1:  Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's Day Off".
  Fact 1.0: Clausen has scored or orchestrated music for more than 30 films and television shows.
  Fact 1.1: Clausen has scored or orchestrated music for 'Moonlighting'.
  Fact 1.2: Clausen has scored or orchestrated music for 'The Naked Gun'.
  Fact 1.3: Clausen has scored or orchestrated music for 'ALF'.
  Fact 1.4: Clausen has scored or orchestrated music for 'Ferris Bueller's Day Off'.

Retrieved items:
<item 1>
Angelo Badalamenti
Angelo Badalamenti (born March 22, 1937) is an American composer, best known for his work scoring films for director David Lynch, notably "Blue Velvet", the
"Twin Peaks" saga (1990–1992, 2017), "The Straight Story" and "Mulholland Drive".

 Badalamenti received the 1990 Grammy Award for Best Pop Instrumental Performance for his ""Twin Peaks" Theme", and has received a Lifetime Achievement Award
from the World Soundtrack Awards and the Henry Mancini Award from the American Society of Composers, Authors and Publishers.
</item 1>

<item 2>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 2>

<item 3>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 3>

<item 4>
Batman (score)
Batman: Original Motion Picture Score is the score album for the 1989 film "Batman" by Danny Elfman.

 According to the "Batman" DVD Special Edition, Elfman said that producer Jon Peters was not sure about him as a composer until Tim Burton made him play the
main titles.

 Elfman admitted he was stunned when Peters announced that the score would be released on its own album, as releasing a separate score album for a film was
something that was rarely done in the 1980s.

 Elfman's "The Batman Theme" went on to become an iconic piece.

 It served as the basis for the theme music of "", which premiered in 1992, although this was later changed.

 Some parts of the Elfman score are also heard in "", "" and "".

 Parts are also played in the queue, and on the station platform of Batman the Ride at various Six Flags theme parks.
</item 4>

<item 5>
The Simpsons Theme
"The Simpsons" Theme", also referred to as "The Simpsons" Main Title Theme" in album releases, is the theme music of the animated television series "The
Simpsons".

 It plays during the opening sequence and was composed by Danny Elfman in 1989, after series creator Matt Groening approached him requesting a retro-style
theme.

 The piece, which took 3 days, 2 hours, 48 minutes, and 19 seconds to create, has been noted by Elfman as the most popular of his career.

 The theme, as used for the opening sequence, was re-arranged during season 2, and the current arrangement by Alf Clausen was introduced at the beginning of the
third season.
</item 5>

<item 6>
James Dooley (composer)
James Michael "Jim" Dooley (born August 22, 1976 in New York City, New York) is an American film score composer.

 Dooley studied music at New York University, majoring in music composition.

 After finishing the university he moved to Los Angeles, where he studied music with prolific film score composers Christopher Young, Elmer Bernstein and
Leonard Rosenman.

 In 1999, he started working for Hans Zimmer as his chief technical assistant.

 He works in Santa Monica, in Hans Zimmer's film music studio Remote Control Productions (formerly "Media Ventures").

 He composed, arranged, and orchestrated music for films like "" and "The Da Vinci Code".

 He also composed music for "inFAMOUS 2", the Epic Mickey series and worked with Celldweller and Tarja Turunen.

 He released his debut album, "Veiled Nation", in 2013.
</item 6>

<item 7>
Alf Clausen
Alf Heiberg Clausen (born March 28, 1941) is an American film and television composer.

 He is best known for his work scoring many episodes of "The Simpsons", of which he had been the sole composer between 1990 and 2017.

 Clausen has scored or orchestrated music for more than 30 films and television shows, including "Moonlighting", "The Naked Gun", "ALF" and "Ferris Bueller's
Day Off".
</item 7>

<item 8>
Disney's Adventures of the Gummi Bears
The show is well-remembered for its theme music, written by Michael and Patty Silversher and creation of "gummiberry juice" which was a type of magic potion,
granting abilities, which allowed them to bounce away from their hunters.

 The theme song was performed by Joseph Williams, son of composer John Williams and one-time lead singer of Toto.
</item 8>

<item 9>
Diwa de Leon
Diwa de Leon is a composer, arranger and musician based in Manila, Philippines.

 He makes music mainly for television shows, films, and stage productions.

 His most notable work is his arrangement of GMA Network's "Survivor Philippines" theme music.

 His songs included in "Emir".

 He won the Cinema One Originals Best Musical Score award for his work on the film "Kolorete" and Best Music Score for "Kamera Obskura" at the 8th Cinemalaya
Independent Film Festival.

 He is also a founding member of the world music group Makiling.

 His first mainstream film project is Mamarazzi produced by Regal Films.
</item 9>

<item 10>
Michelangelo Sosnowitz
Michelangelo Sosnowitz (born August 21, 1973) is an American music composer.

 His works include ballet, musical theater, electronic and popular music, as well as commercial, television and film.

 His feature film scores include A Novel Romance, winner Best Film at the NYC International Independent Film Festival, the Dan Fogler directorial debut
Hysterical Psycho and the Deborah Kampmeier film Split.

 His original ballet 'Petrouchka' won Best Choreography at NYMF 2010.

 He wrote the opening theme music to The History Channel show Battles BC and original theme music for ENTV, Deadline Hollywood BlackTreeMedia and other PMC
programming.

 He also wrote the score and theme music to the Colin Quinn Web series COP SHOW.

 Original Off-Broadway musicals include Crazy Head Space and Love in the MIddle Ages.
</item 10>
=== END ITEM id=73da6115c776c6d0 ===

=== ITEM id=8f2d54b0b97f9f8c ===
Gold sentences and facts:
Sentence 0: The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.
  Fact 0.0: The University of Arizona is a public research university.
  Fact 0.1: The University of Arizona is in Tucson, Arizona, United States.
  Fact 0.2: The University of Arizona is also referred to as U of A.
  Fact 0.3: The University of Arizona is also referred to as UA.
  Fact 0.4: The University of Arizona is also referred to as Arizona.
Sentence 1: The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.
  Fact 1.0: The Presbyterian University College, Ghana is a partially private and public university.
  Fact 1.1: The Presbyterian University College, Ghana is located at Abetifi-Kwahu.
  Fact 1.2: Abetifi-Kwahu is in the Eastern Region of Ghana.
Sentence 2:  It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.
  Fact 2.0: The Presbyterian University College, Ghana is one of the new universities in Ghana.
  Fact 2.1: The Presbyterian University College, Ghana was granted accreditation by the National Accreditation Board.

Retrieved items:
<item 1>
Alpha Gamma
Alpha Gamma (ΑΓ) was an American collegiate fraternity.

 It was founded at Cumberland University in Lebanon, Tennessee in 1867.

 About 21 chapters developed, with prominent chapters at Washington & Jefferson College, Trinity University, Mercersburg College, Southwestern Presbyterian
University, Cumberland University, and the West Virginia University.

 The fraternity's badge consisted of a golden shield with a globe, six stars, and a pennant with "ΑΓ."
</item 1>

<item 2>
University of Arizona
The University of Arizona (also referred to as U of A, UA, or Arizona) is a public research university in Tucson, Arizona, United States.

 Founded in 1885, the UA was the first university in the Arizona Territory.

 The university operates two medical schools (University of Arizona College of Medicine - Tucson and the University of Arizona College of Medicine - Phoenix)
and is affiliated with the region's only academic medical centers (Banner - University Medical Center Tucson and Banner - University Medical Center Phoenix).

 The university is also home to the James E. Rogers College of Law and numerous other nationally ranked graduate and professional schools.

 During the 2016–2017 academic year, there was a total enrollment of 43,625 students, including 34,072 undergraduates The University of Arizona is governed by
the Arizona Board of Regents.

 The University of Arizona is one of the elected members of the Association of American Universities (an organization of North America's premier research
institutions) and is the only representative from the state of Arizona to this group.
</item 2>

<item 3>
Lydia Patterson Institute
Lydia Patterson Institute is a Methodist Christian college-preparatory school located in El Paso, Texas, United States.

 Founded in 1913, it offers programs for Spanish-speaking children, primarily from Juarez, to attend high school in the United States and attend an American
undergraduate university.

 All high school classes are taught in English, and the school is accredited by the Southern Association of Colleges and Schools.
</item 3>

<item 4>
Syracuse Triad
The Syracuse Triad is the name given to the three women's sororities founded at Syracuse University.

 Alpha Phi was founded first in 1872 by 10 of the original 20 women admitted into Syracuse University.

 Gamma Phi Beta came along two years later in 1874 and with it came the term "sorority," which was coined at the time of its founding.

 (Prior to that, women's Greek-letter organizations used the term "women's fraternity," since no more appropriate term existed.)

 Alpha Gamma Delta completed the triad in 1904.
</item 4>

<item 5>
Presbyterian University College
The Presbyterian University College, Ghana is a partially private & public university located at Abetifi-Kwahu in the Eastern Region of Ghana.

 It is one of the new universities in Ghana granted accreditation by the National Accreditation Board.

 It was established by the Presbyterian Church of Ghana on 23 November 2003 and inaugurated on 27 March 2004 by the president of Ghana, John Agyekum Kufuor.
</item 5>

<item 6>
Graph query row:
organization: Presbyterian University College, Ghana
founding_date: 23 November 2003
</item 6>

<item 7>
Gorgias Press
Gorgias Press is an academic publisher of books and journals covering a range of religious and language studies that include Syriac language, Eastern
Christianity, Ancient Near East, Arabic and Islam, Early Christianity, Judaism, and more.

 Gorgias Press was founded in 2001 by George Kiraz, and is based in Piscataway, New Jersey.

 Authors include Sebastian Brock, Clinton Bennett, David C. Parker, Andrei Orlov, Iain Torrance, Philip Khuri Hitti, George Percy Badger, Ignatius Zakka I Iwas,
Ignatius Afram I Barsoum, Ignatius Elias III, Carl Brockelmann, Aziz Suryal Atiya, and William Hatch.
</item 7>

<item 8>
Aaron McCreary
Aaron Monroe "Mac" McCreary (September 15, 1892 – ?)

 was an American football, basketball, and baseball player, coach, and college athletics administrator.

 He served as the head football coach at Tempe State Teachers College, known at Arizona State Teachers College by 1929 and now called Arizona State University,
compiling a career college football record of 25–17–4.

 McCreary was also the head basketball coach at Tempe/Arizona State Teachers from 1923 to 1930 and at Arizona State Teacher's College of Flagstaff, now Northern
Arizona University, amassing a career college basketball record of 140–149.

 In addition, he coached baseball at Tempe/Arizona State Teachers (1924–1926, 1928, 1930–1931) and at Arizona State Teacher's Flagstaff in 1959, tallying a
career college baseball mark of 22–56–1.
</item 8>

<item 9>
Sanders–Brown Center on Aging
The Sanders–Brown Center on Aging at the University of Kentucky in Lexington, Kentucky, is responsible for research, education and service programs in aging.

 The program was founded in 1963 with the creation of the Council on Aging.

 In 1972, a grant from the Eleanor and John Y. Brown Jr., Foundation led to the construction of the current facility.

 The center was named for the Browns and his then-business partner, (Col.) Harlan Sanders.

 The four-story brick and concrete structure fronts South Limestone at the edge of the medical campus.

 The complex features 32 research laboratories that employs more than 150.
</item 9>

<item 10>
Evangelical Presbyterian University College
The Evangelical Presbyterian University College was established in 2008 by the Evangelical Presbyterian Church, Ghana.
</item 10>
=== END ITEM id=8f2d54b0b97f9f8c ===

=== ITEM id=14d9b5c1dceca9a6 ===
Gold sentences and facts:
Sentence 0: Baghdad ER is a documentary released by HBO on May 21, 2006.
  Fact 0.0: Baghdad ER is a documentary.
  Fact 0.1: Baghdad ER was released by HBO.
  Fact 0.2: Baghdad ER was released on May 21, 2006.
Sentence 1: Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium in Berlin, Germany.
  Fact 1.0: Olympia is a documentary film.
  Fact 1.1: Olympia was released in 1938.
  Fact 1.2: Olympia is a German film.
  Fact 1.3: Olympia was written by Leni Riefenstahl.
  Fact 1.4: Olympia was directed by Leni Riefenstahl.
  Fact 1.5: Olympia was produced by Leni Riefenstahl.
  Fact 1.6: Olympia documents the 1936 Summer Olympics.
  Fact 1.7: The 1936 Summer Olympics were held in the Olympic Stadium in Berlin, Germany.

Retrieved items:
<item 1>
Anatomy (film)
Anatomy (German: "Anatomie" ) is a 2000 German horror film written and directed by Stefan Ruzowitzky that stars Franka Potente.

 A sequel, "Anatomy 2" ("Anatomie 2") was released in 2003.

 The film was a box-office success in Germany, and Columbia Pictures released the film's English-dubbed version in the United States theatrically.
</item 1>

<item 2>
List of accolades received by The Hurt Locker
"The Hurt Locker" is a 2009 Iraq War film written by Mark Boal and directed by Kathryn Bigelow.

 The film premiered on September 4, 2008 at the 65th Venice International Film Festival, where it competed with "The Wrestler" for the Golden Lion award.

 It was released in Italy by Warner Bros.

 Pictures on October 10, 2008.

 Following a showing at the 33rd Toronto International Film Festival, Summit Entertainment picked the film up for distribution in the United States.

 "The Hurt Locker" was released in the United States on June 26, 2009, with a limited release at four theaters in Los Angeles and New York City.

 The film grossed over $145,000 on its opening weekend, averaging around $36,000 per theater.

 The following weekend, beginning July 3, the film grossed over $131,000 at nine theaters.

 It held the highest per-screen average of any movie playing theatrically in the United States for the first two weeks of its release, gradually moving into the
top 20 chart.

 Summit Entertainment then released "The Hurt Locker" to more screens.
</item 2>

<item 3>
Kathryn Bigelow
Kathryn Ann Bigelow ( ; born November 27, 1951) is an American director, producer, and writer.

 Her films include the vampire Western horror film "Near Dark" (1987), the action crime film "Point Break" (1991), the science fiction action thriller "Strange
Days" (1995), the mystery thriller "The Weight of Water" (2000), the submarine thriller "" (2002), the war film "The Hurt Locker" (2008), the action thriller
war film "Zero Dark Thirty" (2012), the short film "Last Days" (2014), and the period crime drama "Detroit" (2017).

 "The Hurt Locker" won the 2009 Academy Award for Best Picture and the BAFTA Award for Best Film, and was nominated for the 2010 Golden Globe Award for Best
Drama.

 She has also acted as producer and writer for many of her films.
</item 3>

<item 4>
Baghdad ER
Baghdad ER is a documentary released by HBO on May 21, 2006.

 It shows the Iraq war from the perspective of a military hospital in Baghdad.

 It has some relatively disturbing scenes in it (e.g. amputations), therefore the U.S. Army is officially warning that military personnel watching it could
experience symptoms of post-traumatic stress disorder (PTSD).
</item 4>

<item 5>
Matthew O'Neill (filmmaker)
Matthew O'Neill is a documentary filmmaker best known for his work on the HBO film "Baghdad ER", for which he and co-creator Jon Alpert won three Emmy Awards.
</item 5>

<item 6>
Shamal Sabri
He has proven himself over and over as an important player in Kurdistan's developing film industry and eventually earned the position of Artistic Director of
Duhok IFF in 2016.

 Shamal is the head of submission committee for films representing Iraq in the Asia Pacific Screen Awards.
</item 6>

<item 7>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 7>

<item 8>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 8>

<item 9>
Baghdad Jewish Arabic
Baghdad Jewish Arabic is the Arabic dialect spoken by the Jews of Baghdad and other towns of Southern Iraq.

 This dialect differs from the dialect spoken by the Jews in Northern Iraq, such as Mosul and 'Ana.

 The Baghdadi and Northern dialects may be regarded as subvarieties of Judeo-Iraqi Arabic.

 As with most Judeo-Arab communities, there are likely to be few, if any, speakers of the Judeo-Iraqi Arabic dialects who still reside within Iraq.

 Rather these dialects have been maintained or are facing critical endangerment within respective Judeo-Iraqi diasporas, namely those of Israel and the United
States.

 In 2014, the film "Farewell Baghdad" (Arabic: مطير الحمام; Hebrew: מפריח היונים, lit.

 "The Dove Flyer"), which is performed mostly in Jewish Baghdadi Arabic dialect, became the first film to be almost completely performed in Judeo-Iraqi Arabic.
</item 9>

<item 10>
Olympia (1938 film)
Olympia is a 1938 German documentary film written, directed and produced by Leni Riefenstahl, documenting the 1936 Summer Olympics, held in the Olympic Stadium
in Berlin, Germany.

 The film was released in two parts: "Olympia 1.

 Teil — Fest der Völker" (Festival of Nations) and "Olympia 2.

 Teil — Fest der Schönheit" (Festival of Beauty).

 It was the first documentary feature film of the Olympic Games ever made.

 Many advanced motion picture techniques, which later became industry standards but which were groundbreaking at the time, were employed —including unusual
camera angles, smash cuts, extreme close-ups, placing tracking shot rails within the bleachers, and the like.

 The techniques employed are almost universally admired, but the film is controversial due to its political context.

 Nevertheless, the film appears on many lists of the greatest films of all time, including "Time" magazine's "All-Time 100 Movies."
</item 10>
=== END ITEM id=14d9b5c1dceca9a6 ===

=== ITEM id=ed2a29ec953bfc4c ===
Gold sentences and facts:
Sentence 0: Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.
  Fact 0.0: Frederick King Keller was born in 1954.
  Fact 0.1: Frederick King Keller was born in Buffalo, New York.
  Fact 0.2: Frederick King Keller is an American director.
  Fact 0.3: Frederick King Keller is an American producer.
  Fact 0.4: Frederick King Keller is an American screenwriter.
  Fact 0.5: Frederick King Keller works in film and television.
Sentence 1: Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
  Fact 1.0: Ian Schneider was born on October 3, 1992.
  Fact 1.1: Ian Schneider is an independent American film director.
  Fact 1.2: Ian Schneider is an American screenwriter.

Retrieved items:
<item 1>
Ian Schneider is an independent American director and screenwriter. Mary Sweeney, a longtime David Lynch collaborator, edited and produced *Twin Peaks*, *Lost
Highway*, *The Straight Story* and *Mulholland Drive*; she teaches at USC and chairs Film Independent. Angelo Badalamenti scored Lynch works and won a Grammy.
Brian Ralston, Nancy Schreiber and John Bailey are established composers/cinematographers; Bailey led the Motion Picture Academy. Jon Paul Puno is
</item 1>

<item 2>
Kurt Voss
Kurt Voss (born Kurt Christopher Peter Wössner) is an American film director, screenwriter and musician-songwriter.

 Voss's credits include Will Smith's debut "Where The Day Takes You"; the Justin Theroux, Alyssa Milano and Ice T action film "Below Utopia"; actress Jaime
Pressly's debut feature "", and rock and roll related films including "Down and Out with the Dolls" and "Ghost on The Highway: A Portrait of Jeffrey Lee Pierce
and The Gun Club".
</item 2>

<item 3>
Klete Keller
Klete D. Keller (born March 21, 1982) is an American former competition swimmer who won medals at the 2000 Summer Olympics and the 2004 Summer Olympics in the
400-meter freestyle and the 4×200-meter freestyle relay.

 In the 4×200-meter freestyle relay, Keller held off a charging Ian Thorpe in the anchor leg to win the race by 0.13 seconds.

 This was the first time Australia had been beaten in the event in over seven years.

 The American relay of Michael Phelps, Ryan Lochte, Peter Vanderkaay, and Keller are undefeated since the Athens games.

 Vanderkaay, Larsen Jensen, Erik Vendt, and Keller make up the core of the premier American mid-distance/distance freestyle swimmers.
</item 3>

<item 4>
The entries profile filmmakers, producers, performers, and artists: Tim Bedore is a Chicago-born comedian; Rogue One (2016) depicts rebels stealing Death Star
plans; Frederick K. Keller, Peter Guber, Baltasar Kormákur, Steve Rash, Jonathan Kaplan, Kurt Voss, Jim Sharman, Barry Sandler, and Howard Rosenman are
directors or producers. Other highlights include Titan A.E.’s commercial failure; Vito Russo’s documentary; Scott Free’s founding by the
</item 4>

<item 5>
Colby Keller
Colby Keller (born October 18, 1980) is an American visual artist, blogger, and pornographic film actor.

 His career in porn film started in 2004 at Sean Cody and has since expanded to include such studios as Cocksure Men, Randy Blue, Titan Men, Falcon, CockyBoys
and Men.com.

 Keller has amassed a large fan base with both his films and his long-standing blog, "The Big Shoe Diaries".

 He has been nominated for a number of awards and in 2013, World of Wonder awarded him the WOWie Award for Best Blog.

 Keller is nominated for 8 2015 Grabby Awards.

 Keller has also appeared in the short, Zolushka, a queer retelling of Cinderella and the popular series Capitol Hill both by the cult filmmaker Wes Hurley.
</item 5>

<item 6>
Ian Schneider
Ian Schneider (born October 3, 1992), is an independent American film director and screenwriter.
</item 6>

<item 7>
John Bailey (cinematographer)
John Ira Bailey, A.S.C. (born August 10, 1942) is an American cinematographer and film director best known for his collaborations with directors Paul Schrader,
Lawrence Kasdan, Michael Apted, and Ken Kwapis.

 In August 2017, Bailey was elected President of the Academy of Motion Picture Arts and Sciences for a four-year term.
</item 7>

<item 8>
Frederick King Keller
Frederick King Keller (born 1954 in Buffalo, New York) is an American director, producer and screenwriter for film and television.

 He is also credited under the names Frederick K. Keller, Fred K. Keller and Fred Keller.

 He is the son of actor/screenwriter .

 His father was a television pioneer who produced and directed the first weekly dramatic series seen on television.Besides acting and directing in theater his
father also ran several art house movie theaters in Buffalo which the young Fred became intimately involved with and which formed the root of his cinematic
education.
</item 8>

<item 9>
Allen Klein
Allen Klein (December 18, 1931 July 4, 2009) was an American businessman, music publisher, writers' representative, filmmaker and record label executive, most
noted for his tough persona and aggressive, innovative negotiation tactics, many of which established higher industry standards for compensating recording
artists.

 He founded ABKCO Music & Records Incorporated.

 Klein revolutionized the income potential of recording artists, who previously had been routinely victimized by onerous record company contracts.

 He first scored massive monetary and contractual windfalls for Buddy Knox and Jimmy Bowen, one-hit rockabillies of the late 1950s, then parlayed his early
successes into a position managing Sam Cooke, and eventually managed the Beatles and the Rolling Stones simultaneously, along with many other artists, becoming
one of the most powerful individuals in the music industry during his era.
</item 9>

<item 10>
Dan Ouellette
Giger and Beksinski, and he has been featured in magazines internationally.

 David Bowie commented while looking at Dan's art that he has noticed a strong influence of science fiction on contemporary art.

 Giger saw a different aspect, saying simply "Very bony."

 He grew to adore the cinema of Fellini and later to marvel at Lynch's "Eraserhead".

 Rather than pursue a formal education in the fine arts he chose to study the craft of film making and after college he became a production designer working on
feature films in New York City starting with his work as a production designer for Hal Hartley in 1990 with Trust and then, in 1992, with Simple Men.
</item 10>
=== END ITEM id=ed2a29ec953bfc4c ===

=== ITEM id=2575db95ea79dfeb ===
Gold sentences and facts:
Sentence 0: The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.
  Fact 0.0: The soundtrack for the 1968 film "Romeo and Juliet" was composed by Nino Rota.
  Fact 0.1: The soundtrack for the 1968 film "Romeo and Juliet" was conducted by Nino Rota.
Sentence 1: Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
  Fact 1.0: Romeo and Juliet is a 1968 British-Italian romance film.
  Fact 1.1: Romeo and Juliet is based on the tragic play of the same name.
  Fact 1.2: The play Romeo and Juliet was written between 1591 and 1595.
  Fact 1.3: The play Romeo and Juliet was written by William Shakespeare.

Retrieved items:
<item 1>
The Godfather (soundtrack)
The Godfather is the soundtrack from the film of the same name, released in 1972 by Paramount Records, and in 1991 on compact disc by MCA.

 Unless noted, the cues were composed by Nino Rota and conducted by Carlo Savina (who was credited on the LP, but not the CD).

 The song "I Have but One Heart" is sung by Al Martino, who performed it in the film as character Johnny Fontane.
</item 1>

<item 2>
The Godfather Part II (soundtrack)
The Godfather Part II is the soundtrack from the movie of the same name, released in 1974 by ABC, and 1991 on compact disc by MCA.

 The original score was composed by Nino Rota and conducted by Carmine Coppola, who also provided source music for the film.

 Rota expands upon two of the three main themes from the first film: "The Godfather Waltz" and "Michael's Theme", while "The Love Theme" from the first film
makes a brief appearance during a flashback sequence ("Remember Vito Andolini").

 There are several new themes, including one for Kay (Diane Keaton), and two for young Vito (Robert De Niro): "The Immigrant Theme" and "The Tarantella",
introduced in "A New Carpet".
</item 2>

<item 3>
Giuseppe Verdi (film)
Giuseppe Verdi is a 1938 Italian biographical film directed by Carmine Gallone and starring Fosco Giachetti, Gaby Morlay and Germana Paolieri.

 The film portrays the life of the composer Giuseppe Verdi (1813-1901).

 The casting of Giachetti as Verdi was intended to emphasise the composer's patriotism, as he had recently played patriotic roles in films such as "The White
Squadron".

 The film was made at the Cinecittà Studios in Rome.

 The film is also known by the alternative title The Life of Giuseppe Verdi.
</item 3>

<item 4>
Romeo and Juliet (1968 film soundtrack)
The soundtrack for the 1968 film "Romeo and Juliet" was composed and conducted by Nino Rota.

 It was originally released as a vinyl record, containing nine entries, most notably the song "What Is a Youth", composed by Nino Rota, written by Eugene Walter
and performed by Glen Weston.

 The music score won a Silver Ribbon award of the Italian National Syndicate of Film Journalists in 1968 and was nominated for two other awards (BAFTA Award for
Best Film Music in 1968 and Golden Globe Award for Best Original Score in 1969).
</item 4>

<item 5>
Il trittico
Il trittico ("The Triptych") is the title of a collection of three one-act operas, "Il tabarro", "Suor Angelica", and "Gianni Schicchi", by Giacomo Puccini.

 The work received its world premiere at the Metropolitan Opera on 14 December 1918.
</item 5>

<item 6>
8½
8½ (Italian title: Otto e mezzo ] ) is a 1963 semi-autobiographical comedy-drama film directed by Federico Fellini.

 Co-scripted by Fellini, Tullio Pinelli, Ennio Flaiano, and Brunello Rondi, it stars Marcello Mastroianni as Guido Anselmi, a famous Italian film director.

 Shot in black-and-white by cinematographer Gianni di Venanzo, the film features a soundtrack by Nino Rota with costume and set designs by Piero Gherardi.
</item 6>

<item 7>
Il cappello di paglia di Firenze
Il cappello di paglia di Firenze (literally translated as "The Florentine Straw Hat" but usually titled in English language productions as "The Italian Straw
Hat") is an opera by Nino Rota to an Italian-language libretto by the composer and Ernesta Rota, based on the play "Le chapeau de paille d'Italie" by Eugène
Labiche and Marc Michel.
</item 7>

<item 8>
West Side Story (film)
West Side Story is a 1961 American musical romantic drama film directed by Robert Wise and Jerome Robbins.

 The film is an adaptation of the 1957 Broadway musical of the same name, which in turn was inspired by William Shakespeare's play "Romeo and Juliet".

 It stars Natalie Wood, Richard Beymer, Russ Tamblyn, Rita Moreno, and George Chakiris, and was photographed by Daniel L. Fapp, A.S.C., in Super Panavision 70.

 Released on October 18, 1961 through United Artists, the film received high praise from critics and viewers, and became the second highest grossing film of the
year in the United States.

 The film was nominated for 11 Academy Awards and won 10, including Best Picture (as well as a special award for Robbins), becoming the record holder for the
most wins for a movie musical.
</item 8>

<item 9>
Romeo and Juliet (1968 film)
Romeo and Juliet is a 1968 British-Italian romance film based on the tragic play of the same name (1591–1595) by William Shakespeare.
</item 9>

<item 10>
Amarcord Nino Rota
Amarcord Nino Rota is an album by various artists, recorded as a tribute to composer Nino Rota.
</item 10>
=== END ITEM id=2575db95ea79dfeb ===
