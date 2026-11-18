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

This file has 11 items.

=== ITEM id=d01842bcf3defcf5 ===
Gold sentences and facts:
Sentence 0: German Spitz is used to refer to both a breed of dog and category or type of dog.
  Fact 0.0: German Spitz is used to refer to a breed of dog.
  Fact 0.1: German Spitz is used to refer to a category or type of dog.
Sentence 1:  Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.
  Fact 1.0: Several modern breeds have been developed from the German Spitz.
  Fact 1.1: These modern breeds are either registered as separate breeds.
  Fact 1.2: These modern breeds are either registered as varieties of German Spitz.
Sentence 2:  The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that would later become known as the English Springer Spaniel.
  Fact 2.0: The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels.
  Fact 2.1: Attempts were made to use the term to specify a breed that would later become known as the English Springer Spaniel.

Retrieved items:
<item 1>
American Eskimo Dog
The American Eskimo Dog is a breed of companion dog originating in Germany.

 The American Eskimo is a member of the Spitz family.

 The breed's progenitors were German Spitz, but due to anti-German prejudice during the First World War, it was renamed "American Eskimo Dog".

 Although modern American Eskimos have been exported as German Spitz Gross (or Mittel, depending on the dog's height), the breeds have diverged and the
standards are significantly different.

 In addition to serving as a watchdog and companion, the American Eskimo Dog also achieved a high degree of popularity in the United States in the 1930s and
1940s as a circus performer.
</item 1>

<item 2>
King Charles Spaniel
The King Charles Spaniel (also known as the English Toy Spaniel) is a small dog breed of the spaniel type.

 In 1903, the Kennel Club combined four separate toy spaniel breeds under this single title.

 The other varieties merged into this breed were the Blenheim, Ruby and Prince Charles Spaniels, each of which contributed one of the four colours available in
the breed.
</item 2>

<item 3>
Norfolk Spaniel
The Norfolk Spaniel or Shropshire Spaniel is an extinct breed of dog since the early 20th century.

 It was originally thought to have originated from the work of one of the Dukes of Norfolk, but this theory was disproven after being in doubt during the later
part of the 19th century.

 The term was used to designate springer type spaniels that were neither Sussex nor Clumber Spaniels, and attempts were made to use it to specify a breed that
would later become known as the English Springer Spaniel.
</item 3>

<item 4>
German Spitz (Klein)
The German Spitz Klein is a breed of dog of the German Spitz type.

 They are usually classed as a toy or utility breed.
</item 4>

<item 5>
Picardy Spaniel
The Picardy Spaniel is a breed of dog developed in France for use as a gundog.

 It is related to the Blue Picardy Spaniel, and still has many similarities, but the Picardy Spaniel is the older of the two breeds.

 It is thought to be one of the two oldest continental spaniel breeds and was favoured by the French nobility, remaining popular for hunting after the French
Revolution due to its weather resistant coat that enabled it to hunt in a variety of conditions and terrain.

 However its popularity waned following the influx of English hunting breeds in the early 20th century.

 Slightly smaller than an English Setter but larger than most of its spaniel cousins, it has no major health issues although as with many breeds with pendulous
ears, it can be prone to ear infections.
</item 5>

<item 6>
American Cocker Spaniel
The American Cocker Spaniel is a breed of sporting dog.

 It is a spaniel type dog that is closely related to the English Cocker Spaniel; the two breeds diverged during the 20th century due to differing breed
standards in America and the UK.

 In the United States, the breed is usually called the Cocker Spaniel, while elsewhere in the world, it is called the American Cocker Spaniel in order to
differentiate between it and its English cousin, which was already known as "Cocker Spaniel" before the American variety was created.

 The word "cocker" is commonly held to stem from their use to hunt woodcock in England, while "spaniel" is thought to be derived from the type's origins in
Spain.
</item 6>

<item 7>
Welsh Springer Spaniel
The Welsh Springer Spaniel is a breed of dog and a member of the spaniel family.

 Thought to be comparable to the old Land Spaniel, they are similar to the English Springer Spaniel and historically have been referred to as both the Welsh
Spaniel and the Welsh Cocker Spaniel.

 They were relatively unknown until a succession of victories in dog trials by the breed increased its popularity.

 Following recognition by The Kennel Club in 1902, the breed gained the modern name of Welsh Springer Spaniel.

 The breed's coat only comes in a single colour combination of white with red markings, usually in a piebald pattern.

 Loyal and affectionate, they can become very attached to family members and are wary of strangers.

 Health conditions are limited to those common among many breeds of dog, although they are affected more than average by hip dysplasia and some eye conditions.

 They are a working dog, bred for hunting, and while not as rare as some varieties of spaniel, they are rarer than the more widely known English Springer
Spaniel with which they are sometimes confused.
</item 7>

<item 8>
German Spitz
German Spitz is used to refer to both a breed of dog and category or type of dog.

 Several modern breeds have been developed from the German Spitz, and are either registered as separate breeds or as varieties of German Spitz.

 All the "German Spitz type" dogs are dogs of the Spitz type of German origin.

 The Großspitz, Mittelspitz, and Kleinspitz breeds of "German Spitz type" are also called the German Spitz in English.
</item 8>

<item 9>
Fox Terrier
Fox Terriers are two different breeds of the terrier dog type: the Smooth Fox Terrier and the Wire Fox Terrier.

 Both of these breeds originated in the 19th century from a handful of dogs who are descended from earlier varieties of British terriers, and are related to
other modern white terrier breeds.

 In addition, a number of breeds have diverged from these two main types of fox terrier and have been recognised separately, including the Jack Russell Terrier,
Miniature Fox Terrier and Rat Terrier.

 The Wire and Smooth Fox Terriers share similar characteristics, the main differences being in the coat and markings.

 They have been successful in conformation shows, more prominently in America than their homeland.
</item 9>

<item 10>
English Springer Spaniel
The English Springer Spaniel is a breed of gun dog in the Spaniel family traditionally used for flushing and retrieving game.

 It is an affectionate, excitable breed with a typical lifespan of twelve to fourteen years.

 They are very similar to the Welsh Springer Spaniel and are descended from the Norfolk or Shropshire Spaniels of the mid-19th century; the breed has diverged
into separate show and working lines.

 The breed suffers from average health complaints.

 The show-bred version of the breed has been linked to "rage syndrome", although the disorder is very rare.

 It is closely related to the Welsh Springer Spaniel and very closely to the English Cocker Spaniel; less than a century ago, springers and cockers would come
from the same litter.

 The smaller "cockers" hunted woodcock while the larger littermates were used to flush, or "spring," game.

 In 1902, The Kennel Club recognized the English Springer Spaniel as a distinct breed.

 They are used as sniffer dogs on a widespread basis.

 The term "Springer" comes from the historic hunting role, where the dog would flush (spring) birds into the air.
</item 10>
=== END ITEM id=d01842bcf3defcf5 ===

=== ITEM id=ef662a2651c576d7 ===
Gold sentences and facts:
Sentence 0:  He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems."
  Fact 0.0: The recipient was awarded the 2012 Nobel Prize in Physics.
  Fact 0.1: The recipient shared the 2012 Nobel Prize in Physics jointly with Serge Haroche.
  Fact 0.2: The 2012 Nobel Prize in Physics was awarded for ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems.
Sentence 1:  Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.
  Fact 1.0: Since 2001, Serge Haroche has been a Professor at the Collège de France.
  Fact 1.1: Since 2001, Serge Haroche has held the Chair of Quantum Physics.

Retrieved items:
<item 1>
François Englert
He is the recipient of the 2013 Prince of Asturias Award in technical and scientific research, together with Peter Higgs and the CERN.
</item 1>

<item 2>
List of Nobel laureates in Chemistry
The Nobel Prize in Chemistry (Swedish: "Nobelpriset i kemi" ) is awarded annually by the Royal Swedish Academy of Sciences to scientists in the various fields
of chemistry.

 It is one of the five Nobel Prizes established by the 1895 will of Alfred Nobel, who died in 1896.

 These prizes are awarded for outstanding contributions in chemistry, physics, literature, peace, and physiology or medicine.

 As dictated by Nobel's will, the award is administered by the Nobel Foundation and awarded by a committee that consists of five members elected by the Royal
Swedish Academy of Sciences.

 The first Nobel Prize in Chemistry was awarded in 1901 to Jacobus Henricus van 't Hoff, of the Netherlands.

 Each recipient receives a medal, a diploma and a monetary award prize that has varied throughout the years.

 In 1901, van 't Hoff received 150,782 SEK, which is equal to 7,731,004 SEK in December 2007.

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 2>

<item 3>
François Englert
François Baron Englert (] ; born 6 November 1932) is a Belgian theoretical physicist and 2013 Nobel prize laureate (shared with Peter Higgs).

 He is Professor emeritus at the Université libre de Bruxelles (ULB) where he is member of the Service de Physique Théorique.

 He is also a Sackler Professor by Special Appointment in the School of Physics and Astronomy at Tel Aviv University and a member of the Institute for Quantum
Studies at Chapman University in California.

 He was awarded the 2010 J. J. Sakurai Prize for Theoretical Particle Physics (with Gerry Guralnik, C. R. Hagen, Tom Kibble, Peter Higgs, and Robert Brout), the
Wolf Prize in Physics in 2004 (with Brout and Higgs) and the High Energy and Particle Prize of the European Physical Society (with Brout and Higgs) in 1997 for
the mechanism which unifies short and long range interactions by generating massive gauge vector bosons.

 He has made contributions in statistical physics, quantum field theory, cosmology, string theory and supergravity.
</item 3>

<item 4>
List of Nobel laureates in Physiology or Medicine
Each recipient receives a medal, a diploma and a monetary award that has varied throughout the years.

 In 1901, von Behring received 150,782 SEK, which is equal to 7,731,004 SEK in December 2008.

 In 2013, the prize was awarded to James E. Rothman, Randy W. Schekman and Thomas C. Südhof; they were recognised "after discovering how cells precisely
transport material".

 The award is presented in Stockholm at an annual ceremony on December 10, the anniversary of Nobel's death.
</item 4>

<item 5>
List of female Nobel laureates
The Nobel Prizes are awarded annually by the Royal Swedish Academy of Sciences, the Swedish Academy, the Karolinska Institute, and the Norwegian Nobel Committee
to individuals who make outstanding contributions in the fields of Chemistry, Physics, Literature, Peace, Physiology or Medicine and Economics.

 All but the economics prize were established by the 1895 will of Alfred Nobel, which dictates that the awards should be administered by the Nobel Foundation.

 The Nobel prize in Economics, or The Sveriges Riksbank Prize in Economic Sciences in Memory of Alfred Nobel, was established in 1968 by the Sveriges Riksbank,
the central bank of Sweden, for outstanding contributions in the field of Economics.

 Each prize is awarded by a separate committee; the Royal Swedish Academy of Sciences awards the Prizes in Physics, Chemistry, and Economics, the Swedish
Academy awards the Prize in Literature, the Karolinska Institute awards the Prize in Physiology or Medicine, and the Norwegian Nobel Committee awards the Prize
in Peace.

 Each recipient receives a medal, a diploma and a cash prize that has varied throughout the years.
</item 5>

<item 6>
David J. Wineland
David Jeffrey Wineland (born February 24, 1944) is an American Nobel-laureate physicist at the National Institute of Standards and Technology (NIST) physics
laboratory.

 His work has included advances in optics, specifically laser cooling trapped ions and using ions for quantum computing operations.

 He was awarded the 2012 Nobel Prize in Physics, jointly with Serge Haroche, for "ground-breaking experimental methods that enable measuring and manipulation of
individual quantum systems."
</item 6>

<item 7>
Serge Haroche
Serge Haroche (born 11 September 1944) is a French physicist who was awarded the 2012 Nobel Prize for Physics jointly with David J. Wineland for
"ground-breaking experimental methods that enable measuring and manipulation of individual quantum systems", a study of the particle of light, the photon.

 This and his other works developed laser spectroscopy.

 Since 2001, Haroche is a Professor at the Collège de France and holds the Chair of Quantum Physics.

 In 1971 he defended his doctoral thesis in physics at the University of Paris VI, his research has been conducted under the direction of Claude
Cohen-Tannoudji.
</item 7>

<item 8>
Brian Josephson
Brian David Josephson, {'1': ", '2': ", '3': ", '4': "} (born 4 January 1940), is a Welsh theoretical physicist and professor emeritus of physics at the
University of Cambridge.

 Best known for his pioneering work on superconductivity and quantum tunnelling, he was awarded the Nobel Prize in Physics in 1973 for his prediction of the
Josephson effect, made in 1962 when he was a 22-year-old PhD student at Cambridge University.

 Josephson is the only Welshman to have won a Nobel Prize in Physics.

 He shared the prize with physicists Leo Esaki and Ivar Giaever, who jointly received half the award for their own work on quantum tunnelling.
</item 8>

<item 9>
Duncan Haldane
Frederick Duncan Michael Haldane {'1': ", '2': ", '3': ", '4': "} (born 14 September 1951), known as F. Duncan Haldane, is a British born physicist who is
Eugene Higgins Professor of Physics at the physics department of Princeton University, and a Distinguished Visiting Research Chair at Perimeter Institute for
Theoretical Physics.

 He won the 2016 Nobel Prize in Physics with David J. Thouless and John Michael Kosterlitz.
</item 9>

<item 10>
Nobel Prize in Physics
The Nobel Prize in Physics (Swedish: "Nobelpriset i fysik" ) is a yearly award given by the Royal Swedish Academy of Sciences for those who conferred the most
outstanding contributions for mankind in the field of physics.

 It is one of the five Nobel Prizes established by the will of Alfred Nobel in 1895 and awarded since 1901; the others being the Nobel Prize in Chemistry, Nobel
Prize in Literature, Nobel Peace Prize, and Nobel Prize in Physiology or Medicine.
</item 10>
=== END ITEM id=ef662a2651c576d7 ===

=== ITEM id=5172c0f72ab63d4f ===
Gold sentences and facts:
Sentence 0: The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.
  Fact 0.0: The Celluloid Closet is a 1995 American documentary film.
  Fact 0.1: The Celluloid Closet was directed by Rob Epstein and Jeffrey Friedman.
  Fact 0.2: The Celluloid Closet was written by Rob Epstein and Jeffrey Friedman.
Sentence 1: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.
  Fact 1.0: Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening".
  Fact 1.1: The film Station to Station is made up of 62 one-minute films.

Retrieved items:
<item 1>
The Boxer (Kele Okereke album)
The Boxer is the debut solo album by Kele Okereke, the lead singer of British indie rock band Bloc Party.

 Okereke released the album under the professional name of Kele on 21 June 2010.

 As promotion, he uploaded the songs "Rise" and "Walk Tall" to his personal website on 13 May.

 The first single from "The Boxer" was "Tenderoni", released on 14 June, and the second, "Everything You Wanted", on 16 August.

 The album's third single "On The Lam" was released on 25 October.
</item 1>

<item 2>
Information (Berlin album)
Information is the first album by Berlin, released in 1980 by Vinyl Records.

 It was recorded during the period when Terri Nunn had temporarily left the band to pursue an acting career and Virginia Macolino performed the lead vocals.

 Several songs were written by previous lead singer Toni Childs who went solo prior to the album's release.
</item 2>

<item 3>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 3>

<item 4>
I Shot Andy Warhol
I Shot Andy Warhol is a 1996 American-British independent film about the life of Valerie Solanas and her relationship with the artist Andy Warhol.

 The film marked the debut of Canadian director Mary Harron.

 The film stars Lili Taylor as Valerie, Jared Harris as Andy Warhol, and Martha Plimpton as Valerie's friend Stevie.

 Stephen Dorff plays Warhol superstar Candy Darling.

 John Cale of The Velvet Underground wrote the film's score despite protests from former band member Lou Reed.

 Yo La Tengo plays an anonymous band that is somewhat reminiscent of the group.
</item 4>

<item 5>
Romance of Celluloid
The Romance of Celluloid is a 1937 short black and white documentary film, narrated by Frank Whitbeck, which goes behind the scenes to look at the manufacture
of film and the making of motion pictures.

 The film was the first of the studio's "Romance of Celluloid" series, which also included;
</item 5>

<item 6>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 6>

<item 7>
Station to Station (film)
Station to Station is the 2014 film of Doug Aitken's 2013 "Nomadic Happening", the film is made up of 62 one-minute films.

 It was distributed internationally in 2015 following its release at the Sundance Film Festival.
</item 7>

<item 8>
The Celluloid Closet
The Celluloid Closet is a 1995 American documentary film directed and written by Rob Epstein and Jeffrey Friedman.

 The film is based on Vito Russo's book of the same name first published in 1981 and on lecture and film clip presentations he gave in 1972–1982.

 Russo had researched the history of how motion pictures, especially Hollywood films, had portrayed gay, lesbian, bisexual and transgender characters.
</item 8>

<item 9>
Vito (film)
Vito is a 2011 documentary produced and directed by Jeffrey Schwarz of the Los Angeles-based production company Automat Pictures.

 The film documents the life of Vito Russo, gay activist, film scholar, and author of "The Celluloid Closet".
</item 9>

<item 10>
Love in Motion (song)
"Love in Motion" is the first new material released by the Australian rock synthpop band Icehouse as a 7" vinyl single-only in October 1981 on Regular Records
for the Australian market.

 The band had been known as Flowers until 27 June 1981 after which they changed their name to Icehouse, they had signed to Chrysalis Records and most of
Flowers' material was released under the new name into Europe, UK and US markets.

 "Love in Motion" peaked at #10 on the Australian singles charts.

 The B-side, "Goodnight, Mr. Matthews" was included on the 1982 album "Primitive Man" with "Love in Motion" included on the Chrysalis Records US / European
versions of the album, the UK 1983 version of the album was re-titled "Love in Motion".
</item 10>
=== END ITEM id=5172c0f72ab63d4f ===

=== ITEM id=46db64e4ce8e24ac ===
Gold sentences and facts:
Sentence 0: ", "My Life As A Teenage Robot" and "The Replacements" and he also did the voice of Knotty in "" (1992) and he also did additional voices in films "Fantasia 2000" (1999) and "Sing" (2016)
  Fact 0.0: This person did voice work for My Life As A Teenage Robot.
  Fact 0.1: This person did voice work for The Replacements.
  Fact 0.2: This person did the voice of Knotty in a 1992 work whose title is blank in the sentence.
  Fact 0.3: This person did additional voices in the film Fantasia 2000 (1999).
  Fact 0.4: This person did additional voices in the film Sing (2016).
Sentence 1:  It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.
  Fact 1.0: The film was directed by Garth Jennings.
  Fact 1.1: The film was written by Garth Jennings.
  Fact 1.2: The film was co-directed by Christophe Lourdelet.
  Fact 1.3: The film starred the voices of Matthew McConaughey.
  Fact 1.4: The film starred the voices of Reese Witherspoon.
  Fact 1.5: The film starred the voices of Seth MacFarlane.
  Fact 1.6: The film starred the voices of Scarlett Johansson.
  Fact 1.7: The film starred the voices of John C. Reilly.
  Fact 1.8: The film starred the voices of Taron Egerton.
  Fact 1.9: The film starred the voices of Tori Kelly.

Retrieved items:
<item 1>
Sing (2016 American film)
Sing is a 2016 American 3D computer-animated musical comedy film produced by Illumination Entertainment.

 It was directed and written by Garth Jennings, co-directed by Christophe Lourdelet, and starring the voices of Matthew McConaughey, Reese Witherspoon, Seth
MacFarlane, Scarlett Johansson, John C. Reilly, Taron Egerton, and Tori Kelly.

 The film is about a group of anthropomorphic animals that enter a singing competition, hosted by a koala hoping to save his theater.
</item 1>

<item 2>
Radio Cape Cod
Radio Cape Cod is a 2008 film directed and produced by Andrew Silver.

 It stars Tamzin Outhwaite as a radio interviewer coming to terms with the loss of her husband and a new love entering her life.

 She also must cope with her teenage daughter, played by Tamzin Merchant, who is experiencing her first love.

 The film also stars O. T. Fagbenle.

 The film won several festival awards.
</item 2>

<item 3>
The Jane Austen Book Club (film)
The Jane Austen Book Club is a 2007 American romantic drama film written and directed by Robin Swicord.

 The screenplay, adapted from the 2004 novel of the same name by Karen Joy Fowler, focuses on a book club formed specifically to discuss the six novels written
by Jane Austen.

 As they delve into Austen's literature, the club members find themselves dealing with life experiences that parallel the themes of the books they are reading.
</item 3>

<item 4>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 4>

<item 5>
Deep Dark Canyon
Deep Dark Canyon is a 2012 American independent dramatic thriller film by husband and wife writer/director team Abe Levy and Silver Tree.

 The film stars Ted Levine, Nick Eversman and Spencer Treat Clark.

 The film was originally called "Lawless" and was shot on location in the rural town of Guerneville, California.
</item 5>

<item 6>
Hotline (2014 film)
Hotline is a 2014 documentary feature film written and directed by Tony Shaff.

 The film explores the intense connections that are made between strangers over the telephone, and explores these anonymous conversations people are often too
hesitant to have with the people closest to them.

 The film stars Miss Cleo, Jeff Ragsdale, Jamie Blaine, and Tonya Jone Miller.
</item 6>

<item 7>
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
</item 7>

<item 8>
Girl 6
Girl 6 is a 1996 American comedy-drama film by director Spike Lee about a phone sex operator.

 Theresa Randle played the title character, and playwright Suzan-Lori Parks wrote the screenplay.

 The soundtrack is composed entirely of songs written by Prince.

 The film was screened in the Un Certain Regard section at the 1996 Cannes Film Festival.

 Directors Quentin Tarantino and Ron Silver make cameo appearances as film directors at a pair of interesting auditions.

 It is the first film directed by Lee in which he did not write the screenplay.
</item 8>

<item 9>
Town Without Pity
Town Without Pity (German: Stadt ohne Mitleid) is a 1961 American, Swiss, and West German international co-production drama film directed by Gottfried
Reinhardt.

 Produced by The Mirisch Corporation, the film stars Kirk Douglas, Christine Kaufmann, and E. G. Marshall.

 Coincidentally, this movie came out the same year that John A. Bennett, to this day the last man executed by the U.S. Army, was hanged for raping an
11-year-old girl.
</item 9>

<item 10>
Small Apartments
Small Apartments is a 2012 American comedy film directed by Jonas Åkerlund.

 It tells the story of Franklin Franklin, played by Matt Lucas, who by mistake kills his landlord, Mr. Olivetti, played by Peter Stormare.

 The cast co-stars Dolph Lundgren, Johnny Knoxville, James Caan, Billy Crystal, Juno Temple, Rebel Wilson, Saffron Burrows and Amanda Plummer.

 The screenplay was written by Chris Millis and adapted from his own novella.

 The film premiered at the South by Southwest Film Festival on March 10, 2012.
</item 10>
=== END ITEM id=46db64e4ce8e24ac ===

=== ITEM id=be366c7cfdfdd952 ===
Gold sentences and facts:
Sentence 0:  It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.
  Fact 0.0: The car described in this sentence was driven by Eddie Irvine.
  Fact 0.1: Eddie Irvine was in his second year with the team.
  Fact 0.2: The car described in this sentence was also driven by Luciano Burti.
  Fact 0.3: Luciano Burti had deputised for Eddie Irvine at the 2000 Austrian Grand Prix.
Sentence 1: Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.
  Fact 1.0: Edmund Irvine Jr. was born on 10 November 1965.
  Fact 1.1: Edmund Irvine Jr. is a retired British racing driver.
  Fact 1.2: Edmund Irvine Jr. is from Northern Ireland.

Retrieved items:
<item 1>
Eddie Irvine
Edmund Irvine Jr. (born 10 November 1965) is a retired British racing driver from Northern Ireland.

 He was a Formula One driver between 1993 and 2002, and runner-up in the 1999 World Drivers' Championship, driving for Scuderia Ferrari.
</item 1>

<item 2>
- **2001 Monaco GP (May 27):** Michael Schumacher won for Ferrari; Rubens Barrichello was second and Eddie Irvine third, earning Jaguar its first podium.
- **1997 Argentine GP (Apr 13):** Jacques Villeneuve won from pole; Irvine and Ralf Schumacher completed the podium. It was the 600th World Championship event.
- **1997 Japanese GP (Oct 12):** Michael Schumacher won; Heinz-H
</item 2>

<item 3>
Eddie Irvine (born 10 November 1965) is a retired Northern Irish Formula One driver who raced from 1993 to 2002 and finished runner-up in the 1999 championship
with Ferrari. He drove Jordan’s 1994 and 1995 cars alongside Rubens Barrichello, then Jaguar’s R2 in 2001 and R3 in 2002, partnering Luciano Burti and later
Pedro de la Rosa in 2001. The Jordan 194
</item 3>

<item 4>
1997 Argentine Grand Prix
The 1997 Argentine Grand Prix was a Formula One motor race held at Autódromo Oscar Alfredo Gálvez in Buenos Aires, Argentina on 13 April 1997.

 It was the third race of the 1997 Formula One season and the 600th World Championship event.

 The 72-lap race was won by Williams driver Jacques Villeneuve after he started from pole position.

 Eddie Irvine finished second for the Ferrari team and Jordan driver Ralf Schumacher came in third.
</item 4>

<item 5>
Spyker F1
The Spyker F1 Team, known as the Etihad Aldar Spyker F1 Team for sponsorship reasons was a Formula One team that competed in the 2007 Formula One World
Championship, and was created by Spyker Cars after their buyout of the short-lived Midland F1 (formerly Jordan Grand Prix) team.

 The change to the Spyker name was accompanied by a switch in racing livery from the red and white previously used by Midland, to an orange and silver
scheme—already seen on the Spyker Spyder GT2-R—orange being the national colour and the auto racing colour of the Netherlands.

 At the end of the 2007 season the team was sold and renamed Force India.
</item 5>

<item 6>
Jaguar R2
The Jaguar R2 was the car with which the Jaguar team competed in the Formula One season.

 It was driven by Eddie Irvine, who was in his second year with the team, and Luciano Burti, who had deputised for Irvine at the 2000 Austrian Grand Prix.

 However, the Brazilian was replaced by Pedro de la Rosa after only four Grands Prix.
</item 6>

<item 7>
Jaguar R3
The Jaguar R3 is a Formula One racing car with which Jaguar Racing competed in the 2002 Formula One season.

 The car was launched on January 4, 2002.

 It was driven by Eddie Irvine and Pedro de la Rosa, both retained from 2001.
</item 7>

<item 8>
Jordan 195
The Jordan 195 was a Formula One car for the 1995 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 The team never employed a test driver.

 The engine was a Peugeot A10 3.0 V10.

 The team's main sponsor was Total.
</item 8>

<item 9>
Jordan 194
The Jordan 194 was a Formula One car for the 1994 season.

 The number 14 seat was taken by Rubens Barrichello and the number 15 seat was taken by Eddie Irvine.

 Kelvin Burt was named as test driver, but his mileage in the car was limited.

 The engine was a Hart 1035 3.5 V10, a developed version of the engine which had proved promising in 1993.

 The team's main sponsor was Sasol, with additional financial support coming from Arisco, the Irish government and other smaller sponsors.

 With driver aids such as traction control and active suspension banned for the 1994 season, the car used a conventional semi-automatic gearbox and pushrod
double wishbone suspension.

 The simpler technical regulations of 1994 seemed to benefit Jordan and the 194 recaptured the level of performance seen in the team's debut season, 1991.

 Indeed, the cut-down airbox and drooping front wing of the 194 recalled the design features of the 1991 car.
</item 9>

<item 10>
1983 Volvo International – Doubles
Sherwood Stewart and Ferdi Taygan were the defending champions but they competed with different partners that year, Stewart with Mark Edmondson and Taygan with
Cássio Motta.
</item 10>
=== END ITEM id=be366c7cfdfdd952 ===

=== ITEM id=812db96a18f7b3f4 ===
Gold sentences and facts:
Sentence 0: Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.
  Fact 0.0: Buried Country is a highly regarded documentary film
  Fact 0.1: Buried Country is a highly regarded book
  Fact 0.2: Buried Country is a highly regarded soundtrack album
  Fact 0.3: Buried Country is now also a stageshow
Sentence 1: South of the Border is a 2009 American documentary film directed by Oliver Stone.
  Fact 1.0: South of the Border is a documentary film
  Fact 1.1: South of the Border is American
  Fact 1.2: South of the Border is from 2009
  Fact 1.3: South of the Border was directed by Oliver Stone

Retrieved items:
<item 1>
The Colleen Bawn
The Colleen Bawn, or The Brides of Garryowen is a melodramatic play written by Irish playwright Dion Boucicault.

 It was first performed at Miss Laura Keene's Theatre, New York, on 27 March 1860 with Laura Keene playing Anne Chute and Boucicault playing Myles na Coppaleen.

 It was most recently performed in Dublin at the Project Arts Centre in July and August 2010.

 Several film versions have also been made.
</item 1>

<item 2>
Buried Country
Buried Country is a highly regarded documentary film, book and soundtrack album and now also a stageshow as well.

 It is a prosopography, created by noted writer Clinton Walker, that tells the story of Australian country music in the Aboriginal community by focussing on the
genre's most important stars.
</item 2>

<item 3>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 3>

<item 4>
South of the Border (2009 film)
South of the Border is a 2009 American documentary film directed by Oliver Stone.

 The documentary premiered at the 2009 Venice Film Festival.

 Writer for the project Tariq Ali calls the documentary "a political road movie".

 Stone stated that he hopes the film will help people better understand a leader who is wrongly ridiculed "as a strongman, as a buffoon, as a clown."
</item 4>

<item 5>
Simon Bent
Simon Bent is a British screenwriter and playwright, notable for work including BBC TV drama "" (2006), the screenplay for the feature film "Christie Malry's
Own Double-Entry" (2000), and the Joe Orton biographical play "Prick Up Your Ears" based on John Lahr's book.
</item 5>

<item 6>
Plain Clothes Theatre Productions
Plain Clothes Theatre Productions is a Bristol-based theatre company producing contemporary plays from around the globe.

 Formed in 2003 by artistic director Sam Berger, the company has produced work in London, Bristol, Cheltenham, Toronto and Vancouver.

 Their work has included plays by Joe Orton, Laura Wade and David Mamet, and their style is based around the teachings of American acting coach Sanford Meisner.
</item 6>

<item 7>
Marisha Chamberlin
Marisha Anne Chamberlain (born January 6, 1952) is an American writer.

 Her most recent work, the libretto for "Mortals & Angels", a collaboration with American composer Carol Barnett, premiered at Carnegie Hall in 2016, a
companion piece to their widely produced collaborative piece, "The World Beloved: A Bluegrass Mass".

 Her stage plays, both original works and adaptations are widely produced.

 Her debut novel, "The Rose Variations", was published by Soho Press in 2009.

 Her play, "Scheherazade", won the Dramatists Guild/CBS National Award, and in her screenplay version, played on public television across the country, and was
screened at the British Film Institute Festival in the category of Best of American Public Television.
</item 7>

<item 8>
Cinema of Uruguay
The cinema of Uruguay has a role in the culture of Uruguay and is a part of Latin American cinema.

 Since the late 1990s, the Uruguayan cinema undergoes a process of evolution, during which its films have received positive reviews and been internationally
recognized.

 Over 120 films, fictions and non-fictions, have been produced since then.
</item 8>

<item 9>
Bruce Rowland
Bruce Rowland (born May 9, 1942 in Melbourne) is a well-known Australian composer.

 He composed the soundtrack for the 1982 movie "The Man from Snowy River", as well as the soundtrack for its 1988 sequel "The Man from Snowy River II" (which
has the United States title of "Return to Snowy River", and the United Kingdom title of "The Untamed").

 Both films were based on Banjo Paterson's poem The Man from Snowy River.

 His other film scores include "Now and Forever" (1983), "Phar Lap" (1983), "Bushfire Moon" (1987), "Cheetah" (1989), "Weekend with Kate" (1990), "Gross
Misconduct" (1993), "Andre" (1994), "Lightning Jack" (1994), "Zeus and Roxanne" (1997) and the TV movie "" (1997).
</item 9>

<item 10>
In Country
In Country is a 1989 American drama film produced and directed by Norman Jewison, starring Bruce Willis and Emily Lloyd.

 The screenplay by Frank Pierson and Cynthia Cidre was based on the novel by Bobbie Ann Mason.

 The original music score was composed by James Horner.

 Willis earned a best supporting actor Golden Globe nomination for his role.
</item 10>
=== END ITEM id=812db96a18f7b3f4 ===

=== ITEM id=215b4c4e006c8dc4 ===
Gold sentences and facts:
Sentence 0: Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous family.
  Fact 0.0: Homer Jay Simpson is a fictional character.
  Fact 0.1: Homer Jay Simpson is the main protagonist of the American animated television series 'The Simpsons'.
  Fact 0.2: Homer Jay Simpson is the patriarch of the Simpson family, the family the show is named after.
Sentence 1:  He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April 19, 1987.
  Fact 1.0: Homer Simpson is voiced by Dan Castellaneta.
  Fact 1.1: Homer Simpson first appeared on television along with the rest of his family.
  Fact 1.2: Homer Simpson's first television appearance was in the short 'Good Night' from 'The Tracey Ullman Show'.
  Fact 1.3: Homer Simpson's first television appearance was on April 19, 1987.
Sentence 2: Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.
  Fact 2.0: Daniel Louis Castellaneta was born October 29, 1957.
  Fact 2.1: Daniel Louis Castellaneta is an American actor.
  Fact 2.2: Daniel Louis Castellaneta is a voice actor.
  Fact 2.3: Daniel Louis Castellaneta is a comedian.
  Fact 2.4: Daniel Louis Castellaneta is a screenwriter.

Retrieved items:
<item 1>
Dan Castellaneta
Daniel Louis Castellaneta ( ; born October 29, 1957) is an American actor, voice actor, comedian and screenwriter.

 Noted for his long-running role as Homer Simpson on the animated television series "The Simpsons", he also voices many other characters for the show, including
Abraham "Grampa" Simpson, Barney Gumble, Krusty the Clown, Sideshow Mel, Groundskeeper Willie, Mayor Quimby and Hans Moleman.
</item 1>

<item 2>
Ned Flanders
Nedward "Ned" Flanders, Jr. is a recurring fictional character in the animated television series "The Simpsons".

 He is voiced by Harry Shearer, and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 He is the good-natured, cheery next-door neighbor to the Simpson family and is generally loathed by Homer Simpson.

 A devout Evangelical Christian with an annoyingly perfect family, he is among the friendliest and most compassionate of Springfield's residents and is
generally considered a pillar of the Springfield community.
</item 2>

<item 3>
Homer to the Max
"Homer to the Max" is the thirteenth episode of "The Simpsons"<nowiki>'</nowiki> tenth season.

 It originally aired on the Fox network in the United States on February 7, 1999.

 In the episode, Homer discovers that a new television show, "Police Cops", has a hero also named Homer Simpson.

 He is delighted with the positive attention he receives because of his name, but when the television character is rewritten from a hero to a bumbling idiot, he
is mocked and taunted, so he changes his name to "Max Power" to rid himself of the negative attention.

 Max gains new friends, and is forced into a protest to prevent a forest from being knocked down.

 In the end, he changes his name back to Homer Simpson.
</item 3>

<item 4>
Homer Simpson, This Is Your Wife
"Homer Simpson, This Is Your Wife" is the fifteenth episode of the seventeenth season of the American animated television sitcom "The Simpsons".

 It originally aired on the Fox network in the United States on March 26, 2006, and was watched by around ten million people during that broadcast.

 In the episode, Homer signs the Simpson family up for a reality show in which the mothers of two families switch places.

 Marge gets to live with a friendly man named Charles and his perfect son, while Homer, Bart, and Lisa must spend time with Charles' strict wife Verity.
</item 4>

<item 5>
Homer Simpson
Homer Jay Simpson is a fictional character and the main protagonist of the American animated television series "The Simpsons" as the patriarch of the eponymous
family.

 He is voiced by Dan Castellaneta and first appeared on television, along with the rest of his family, in "The Tracey Ullman Show" short "Good Night" on April
19, 1987.

 Homer was created and designed by cartoonist Matt Groening while he was waiting in the lobby of James L. Brooks' office.

 Groening had been called to pitch a series of shorts based on his comic strip "Life in Hell" but instead decided to create a new set of characters.

 He named the character after his father, Homer Groening.

 After appearing for three seasons on "The Tracey Ullman Show", the Simpson family got their own series on Fox that debuted December 17, 1989.
</item 5>

<item 6>
Citizen Joe
"Citizen Joe" is the fifteenth episode for season eight of the Canadian-American military science fiction television series "Stargate SG-1".

 The episode features known voice actor Dan Castellaneta, who voices Homer Simpson in "The Simpsons".

 The episode was written by executive producer Robert C. Cooper, the episode was directed by Andy Mikita.

 The episode received a below average Nielsen household rating and received no syndication rating to compare.

 The episode got strong reviews from major media publishers worldwide.
</item 6>

<item 7>
Moe Szyslak
Morris "Moe" Szyslak is a fictional character from the American animated television series "The Simpsons".

 He is voiced by Hank Azaria and first appeared in the series premiere episode "Simpsons Roasting on an Open Fire".

 Moe is the proprietor and bartender of Moe's Tavern, a Springfield bar frequented by Homer Simpson, Barney Gumble, Carl Carlson, Lenny Leonard, Sam, Larry, and
others.
</item 7>

<item 8>
The Itchy &amp; Scratchy Show
(voiced by Alex Rocco, and, later, Hank Azaria), who runs the studio and produces the show.
</item 8>

<item 9>
Grampa Simpson
Abraham Jedediah Simpson II, often known as Grampa, is a fictional character in the animated television series "The Simpsons".

 He made his first appearance in the episode entitled "Grampa and the Kids", a Simpsons short on "The Tracey Ullman Show".

 Voiced by Dan Castellaneta, he is the father of Homer Simpson and the grandfather of Bart, Lisa and Maggie Simpson.

 In the 1000th issue of "Entertainment Weekly", Grampa was selected as the Grandpa for "The Perfect TV Family".
</item 9>

<item 10>
Mona Simpson (The Simpsons)
Mona Penelope Simpson (née Olsen) is a recurring fictional character in the animated television series "The Simpsons".

 She has been voiced by several actresses, including Maggie Roswell, Tress MacNeille, Pamela Hayden, and most prominently, Glenn Close.

 Glenn Close's performances as Mona have been well received by critics and she was named one of the top 25 guest stars on the show by IGN.
</item 10>
=== END ITEM id=215b4c4e006c8dc4 ===

=== ITEM id=e532ccb413d6ea9d ===
Gold sentences and facts:
Sentence 0: Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.
  Fact 0.0: Edward Benjamin Britten, Baron Britten, lived from 22 November 1913 to 4 December 1976.
  Fact 0.1: Edward Benjamin Britten, Baron Britten, was an English composer.
  Fact 0.2: Edward Benjamin Britten, Baron Britten, was an English conductor.
  Fact 0.3: Edward Benjamin Britten, Baron Britten, was an English pianist.
Sentence 1: Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.
  Fact 1.0: Béla Viktor János Bartók lived from 25 March 1881 to 26 September 1945.
  Fact 1.1: Béla Viktor János Bartók was a Hungarian composer.
  Fact 1.2: Béla Viktor János Bartók was a Hungarian pianist.
  Fact 1.3: Béla Viktor János Bartók was a Hungarian ethnomusicologist.

Retrieved items:
<item 1>
Benjamin Britten
Edward Benjamin Britten, Baron Britten {'1': ", '2': ", '3': ", '4': "} (22 November 1913 – 4 December 1976) was an English composer, conductor and pianist.

 He was a central figure of 20th-century British classical music, with a range of works including opera, other vocal music, orchestral and chamber pieces.

 His best-known works include the opera "Peter Grimes" (1945), the "War Requiem" (1962) and the orchestral showpiece "The Young Person's Guide to the Orchestra"
(1945).
</item 1>

<item 2>
List of string quartets by Béla Bartók
1 and 28 both owe a great deal to Bartók's quartets (; ), György Ligeti, whose two string quartets both owe a great deal to Bartók’s quartets (; ), Bruno
Maderna , George Perle, who credits the Bartók Fourth and Fifth Quartets as precedents for his use of arrays of chords related to one another by different types
of symmetry , Walter Piston (; ), Kim Dzmitrïyevich Tsesakow , Wilfried Westerlinck , Stefan Wolpe, who explained in a public lecture how he had derived ideas
from Bartók’s Fourth Quartet , and Xu Yongsan .
</item 2>

<item 3>
String Quartet No. 6 (Bartók)
The String Quartet No. 6, Sz.

 114, BB 119, was the final string quartet that Béla Bartók wrote before his death.

 It was begun in August 1939 in Saanen, Switzerland, where Bartók was a guest of his patron, the conductor Paul Sacher.

 Shortly after he completed the Divertimento for String Orchestra on the 17th, he started on a commission for his friend, the violinist Zoltán Székely.

 Székely was acting as intermediary for the "New Hungarian Quartet", who had given the Budapest premiere of the String Quartet No. 5.

 With the outbreak of World War II and his mother's illness, Bartók returned to Budapest, where the quartet was finished in November.

 After his mother's death, Bartók decided to leave with his family for the United States.
</item 3>

<item 4>
List of string quartets by Béla Bartók
The Hungarian composer Béla Bartók wrote six string quartets, each for the usual forces of two violins, viola and cello.

 Notable composers who have been influenced by them include Benjamin Britten, particularly in the Sonata in C for Cello and Piano (; ), Elliott Carter, who
refers in the opening of his own First String Quartet to Bartók’s Sixth Quartet , Chen Yi , Edison Denisov, whose Second Quartet is closely related to Bartók’s
Fifth Quartet , Franco Donatoni, who was deeply impressed when he heard a broadcast of Bartók's Fourth Quartet , Robert Fripp, who mentions them as an influence
upon King Crimson , Miloslav Ištvan , György Kurtág, whose Opp.
</item 4>

<item 5>
Bluebeard's Castle
Bluebeard's Castle (Hungarian: A kékszakállú herceg vára ; literally: "The Blue-Bearded Duke's Castle") is a one-act opera by Hungarian composer Béla Bartók.

 The libretto was written by Béla Balázs, a poet and friend of the composer, and is written in Hungarian, based on the French literary tale "La Barbe bleue" by
Charles Perrault.

 The opera lasts only a little over an hour and there are only two singing characters onstage: Bluebeard ("Kékszakállú "), and his new wife Judith ("Judit " );
the two have just eloped and Judith is coming home to Bluebeard's castle for the first time.
</item 5>

<item 6>
Béla Bartók
Béla Viktor János Bartók ( ; ] ; 25 March 1881 – 26 September 1945) was a Hungarian composer, pianist and an ethnomusicologist.

 He is considered one of the most important composers of the 20th century; he and Liszt are regarded as Hungary's greatest composers .

 Through his collection and analytical study of folk music, he was one of the founders of comparative musicology, which later became ethnomusicology.
</item 6>

<item 7>
List of compositions by Béla Bartók
This aspires to be a complete list of compositions by Béla Bartók.

 The catalogue numbering by András Szőllősy (Sz.)

, László Somfai (BB.)

 and Denijs Dille (DD.)

 are provided, as well as Bartók's own opus numbers.

 Note that Bartók started three times anew with opus numbers, here indicated with "(list 1)", "(list 2)" and "(list 3)" respectively.

 The pieces from the third listing are by far best known; opus lists 1 and 2 are early works.

 The year of composition and instrumentation (including voice) are included.

 See the main article on Béla Bartók for more details.
</item 7>

<item 8>
String Quartet No. 6 (Bartók)
Due to the difficulties of the war, communication between Bartók and Székely was difficult, and the quartet was not premiered until 20 January 1941, when the
Kolisch Quartet, to whom the work is dedicated, gave its premiere at the Town Hall in New York City.
</item 8>

<item 9>
Béla Bartók Music High School
Béla Bartók Music High School (Bartók Béla Zeneművészeti Szakközépiskola) is situated in the Palace of Music (Zenepalota) in Bartók square Miskolc, Hungary.

It is a famous Music school named after the famous Hungarian composer Béla Bartók.

It was founded in 1966.
</item 9>

<item 10>
Suite, Op. 14 (Bartók)
The Suite, Op. 14, Sz.

 62, BB 70 is a piece for solo piano written by Béla Bartók.

 It was written in February 1916, published in 1918, and debuted by the composer on April 21, 1919, in Budapest.

 The Suite is one of Bartók's most significant works for piano, only comparable with his 1926 Piano Sonata.

 Though much of Bartók's work makes frequent use of Eastern European folk music, this suite is one of the few pieces without melodies of folk origin.

 However, Romanian, Arabic, and North African rhythmic influences can still be found in some movements.

 Originally intending the suite to be a five-movement work, Bartók later decided against the idea and discarded the second movement, the "Andante", which was
published only posthumously in the October 1955 issue of "Új Zenei Szemle" (New Musical Review).
</item 10>
=== END ITEM id=e532ccb413d6ea9d ===

=== ITEM id=0f0acf29d374ce41 ===
Gold sentences and facts:
Sentence 0: Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11, 1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical physics.
  Fact 0.0: Nikolay Mitrofanovich Krylov was born on 29 November [O.S. 17 November] 1879.
  Fact 0.1: Nikolay Mitrofanovich Krylov died on May 11, 1955.
  Fact 0.2: Nikolay Mitrofanovich Krylov's Russian name is Николай Митрофанович Крылов.
  Fact 0.3: Nikolay Mitrofanovich Krylov's Ukrainian name is Микола Митрофанович Крилов.
  Fact 0.4: Nikolay Mitrofanovich Krylov was a Russian mathematician.
  Fact 0.5: Nikolay Mitrofanovich Krylov was a Soviet mathematician.
  Fact 0.6: Nikolay Mitrofanovich Krylov is known for works on interpolation.
  Fact 0.7: Nikolay Mitrofanovich Krylov is known for works on non-linear mechanics.
  Fact 0.8: Nikolay Mitrofanovich Krylov is known for works on numerical methods for solving equations of mathematical physics.
Sentence 1: Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.
  Fact 1.0: Anatoly Timofeevich Fomenko's Russian name is Анатолий Тимофеевич Фоменко.
  Fact 1.1: Anatoly Timofeevich Fomenko was born on 13 March 1945 in Stalino, USSR.
  Fact 1.2: Anatoly Timofeevich Fomenko is a Soviet mathematician.
  Fact 1.3: Anatoly Timofeevich Fomenko is a Russian mathematician.
  Fact 1.4: Anatoly Timofeevich Fomenko is a professor at Moscow State University.
  Fact 1.5: Anatoly Timofeevich Fomenko is well known as a topologist.
  Fact 1.6: Anatoly Timofeevich Fomenko is a member of the Russian Academy of Sciences.

Retrieved items:
<item 1>
Anatoly Maltsev
Anatoly Ivanovich Maltsev (also: Malcev, Mal'cev; Russian: Анато́лий Ива́нович Ма́льцев; 27 November N.S./14 November O.S. 1909, Moscow Governorate – 7 June
1967, Novosibirsk) was born in Misheronsky, near Moscow, and died in Novosibirsk, USSR.

 He was a mathematician noted for his work on the decidability of various algebraic groups. Malcev algebras (generalisations of Lie algebras) are named after
him.
</item 1>

<item 2>
Lev Kuleshov
Lev Vladimirovich Kuleshov (Russian: Лев Влади́мирович Кулешо́в ; 13 January [O.S. 1 January] 1899 – 29 March 1970) was a Russian and Soviet filmmaker and film
theorist, one of the founders of the world's first film school, the Moscow Film School.

 He was given the title People's Artist of the RSFSR in 1969.

 He was intimately involved in development of the style of film making known as Soviet montage, especially it's psychological underpinning, including the use of
editing and the cut to emotionally influence the audience, a principle known as the Kuleshov effect.

 He also developed the theory of creative geography, which is the use of the action around a cut to connect otherwise disparate settings into a cohesive
narrative.
</item 2>

<item 3>
Martemyan Ryutin
Martemyan Nikitich Ryutin (Russian: Мартемья́н Ники́тич Рю́тин ) (1890–1937) was a Russian Marxist revolutionary, and a political functionary of the Russian
Communist Party.

 Ryutin is best remembered as the leader of a pro-peasant political faction organized against Soviet leader Joseph Stalin in the early 1930s and as the primary
author of a 200-page oppositional platform.

 Ryutin was arrested by the Soviet secret police, along with his co-thinkers, in what has come to be known as the Ryutin Affair.

 He was executed in January 1937 as part of the Yezhovshchina (Great Purge) conducted against political oppositionists and suspected economic "wreckers" and
spies.
</item 3>

<item 4>
Vladimir Arnold
Vladimir Igorevich Arnold (alternative spelling Arnol'd, Russian: Влади́мир И́горевич Арно́льд , 12 June 1937 – 3 June 2010) was a Soviet and Russian
mathematician.

 While he is best known for the Kolmogorov–Arnold–Moser theorem regarding the stability of integrable systems, he made important contributions in several areas
including dynamical systems theory, catastrophe theory, topology, algebraic geometry, symplectic geometry, differential equations, classical mechanics,
hydrodynamics and singularity theory, including posing the ADE classification problem, since his first main result—the solution of Hilbert's thirteenth problem
in 1957 at the age of 19.
</item 4>

<item 5>
Semyon Aranovich Gershgorin
Semyon Aronovich Gershgorin (August 24, 1901 – May 30, 1933) was a Soviet (born in Pruzhany, Belarus, Russian Empire) mathematician.

 He began as a student at the Petrograd Technological Institute in 1923, became a Professor in 1930, and was given an appointment at the Leningrad Mechanical
Engineering Institute in the same year.

 His contributions include the Gershgorin circle theorem.
</item 5>

<item 6>
Nicolai V. Krylov
Bellman equations), which were examined with stochastic methods.

 This led to the Evans-Krylov theory, for which he received with Lawrence C. Evans in 2004 the Leroy P. Steele Prize of the American Mathematical Society (for
work done simultaneously and independently by both Krylov and Evans).

 They proved the second order differentiability (Hölder continuity of the second derivative) of the solutions of convex, completely nonlinear, second order
elliptical partial differential equations and thus the existence of "classical solutions" (Theorem of Evans-Krylov).

 He was in 1978 at Helsinki and in 1986 at Berkeley an Invited Speaker for the ICM.

 He received the Humboldt Research Award in 2001.

 In 1993 he was elected a member of the American Academy of Arts and Sciences (1993).

 He should not be confused with the mathematician Nikolay M. Krylov.
</item 6>

<item 7>
Nicolai V. Krylov
Nicolai Vladimirovich Krylov (Russian: Никола́й Влади́мирович Крыло́в ; born 5 June 1941) is a Russian mathematician specializing in partial differential
equations, particularly stochastic partial differential equations and diffusion processes.

 Krylov studied at Lomonosov University, where he in 1966 under E. B. Dynkin attained a doctoral candidate title (similar to a PhD) and in 1973 a Russian
doctoral degree (somewhat more prestigious than a PhD).

 He taught from 1966 to 1990 at the Lomonosov University and is since 1990 a professor at the University of Minnesota.

 At the beginning of his career (starting from 1963) he, in collaboration with Dynkin, worked on nonlinear stochastic control theory, making advances in the
study of convex, nonlinear partial equations of 2nd order ("i.e." Bellman equations), which were examined with stochastic methods.
</item 7>

<item 8>
Anatoly Lunacharsky
Anatoly Vasilyevich Lunacharsky (Russian: Анато́лий Васи́льевич Лунача́рский , , 23 November [O.S. 11 November] 1875 – 26 December 1933) was a Russian Marxist
revolutionary and the first Soviet People's Commissar of Education responsible for culture and education.

 He was active as an art critic and journalist throughout his career.
</item 8>

<item 9>
Nikolay Mitrofanovich Krylov
Nikolay Mitrofanovich Krylov (Russian: Никола́й Митрофа́нович Крыло́в , Ukrainian: Микола Митрофанович Крилов ) (29 November [O.S. 17 November] 1879 – May 11,
1955) was a Russian and Soviet mathematician known for works on interpolation, non-linear mechanics, and numerical methods for solving equations of mathematical
physics.
</item 9>

<item 10>
Anatoly Fomenko
Anatoly Timofeevich Fomenko (Russian: Анато́лий Тимофе́евич Фоме́нко ) (born 13 March 1945 in Stalino, USSR) is a Soviet and Russian mathematician, professor at
Moscow State University, well known as a topologist, and a member of the Russian Academy of Sciences.

 He is author of a pseudoscientific theory known as New Chronology.

 He is also a member of the Russian Academy of Natural Sciences (1991).
</item 10>
=== END ITEM id=0f0acf29d374ce41 ===

=== ITEM id=5e84b951c56ae38b ===
Gold sentences and facts:
Sentence 0:  One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
  Fact 0.0: Updike was one of only three writers to win the Pulitzer Prize for Fiction more than once.
  Fact 0.1: The other two writers who won the Pulitzer Prize for Fiction more than once were Booth Tarkington and William Faulkner.
  Fact 0.2: Updike published more than twenty novels during his career.
  Fact 0.3: Updike published more than a dozen short-story collections during his career.
  Fact 0.4: Updike published poetry during his career.
  Fact 0.5: Updike published art and literary criticism during his career.
  Fact 0.6: Updike published children's books during his career.
Sentence 1:  Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.
  Fact 1.0: Seventeen of Updike's novels were bestsellers.
  Fact 1.1: More than 100 million copies of Updike's books are in print.

Retrieved items:
<item 1>
Tom Clancy's Rainbow 6: Patriots
Tom Clancy's Rainbow 6: Patriots is a cancelled first-person shooter video game, part of the "Tom Clancy's Rainbow Six" series, announced on the cover of the
December 2011 issue of "Game Informer".

 It was to be published by Ubisoft, and was developed by the company's Montreal studio, with additional development by Ubisoft Toronto and Red Storm
Entertainment.

 Due to the death of Tom Clancy in October 2013, concern was raised that this game would become the last to bear his name.

 Ubisoft has since stated that they will continue putting Tom Clancy's name on future Tom Clancy titles out of respect for the late author.
</item 1>

<item 2>
John Updike
John Hoyer Updike (March 18, 1932 – January 27, 2009) was an American novelist, poet, short story writer, art critic, and literary critic.

 One of only three writers to win the Pulitzer Prize for Fiction more than once (the others were Booth Tarkington and William Faulkner), Updike published more
than twenty novels, more than a dozen short-story collections, as well as poetry, art and literary criticism and children's books during his career.
</item 2>

<item 3>
Tom Clancy
Thomas Leo "Tom" Clancy Jr. (April 12, 1947 – October 1, 2013) was an American novelist best known for his technically detailed espionage and military-science
story lines set during and after the Cold War.

 Seventeen of his novels were bestsellers, and more than 100 million copies of his books are in print.

 His name was also used on movie scripts written by ghost writers, nonfiction books on military subjects, and video games.

 He was a part-owner of the Baltimore Orioles and vice-chairman of their community activities and public affairs committees.
</item 3>

<item 4>
Joe McGinniss
Joseph R. McGinniss, Sr. (December 9, 1942 – March 10, 2014), known as Joe McGinniss, was an American non-fiction writer and novelist.

 The author of twelve books, he first came to prominence with the best-selling "The Selling of the President 1968" which described the marketing of
then-presidential candidate Richard Nixon.

 He is popularly known for his trilogy of bestselling true crime books — "Fatal Vision", "Blind Faith" and "Cruel Doubt" — which were adapted into TV miniseries
in the 1980s and 90s.

 His last book was "The Rogue: Searching for the Real Sarah Palin", an account of Sarah Palin, the former governor of Alaska who was the 2008 Republican
vice-presidential nominee.
</item 4>

<item 5>
*The 39 Clues: Doublecross*, the series’ fourth installment, follows a divided Cahill family. Ian Kabra’s leadership is challenged by a coup led by the Outcast,
who forces the family to avert historical disasters; revelations include Grace’s cruelty and the Outcast’s identity as Nathaniel Hartford.

*Net Force Explorers* is a Tom Clancy–Steve Pieczenik young-adult spin-off; titles listed include *Deathworld*, *Safe House*,
</item 5>

<item 6>
David Ignatius
David R. Ignatius (May 26, 1950), is an American journalist and novelist.

 He is an associate editor and columnist for "The Washington Post".

 He also co-hosts PostGlobal, an online discussion of international issues at Washingtonpost.com, with Fareed Zakaria.

 He has written nine novels, including "Body of Lies", which director Ridley Scott adapted into a film.

 He is a former Adjunct Lecturer at the Kennedy School of Government at Harvard University and currently Senior Fellow to the Future of Diplomacy Program.

 He has received numerous honors, including the Legion of Honor from the French Republic, the Urbino World Press Award from the Italian Republic, and a lifetime
achievement award from the International Committee for Foreign Journalism.
</item 6>

<item 7>
The entries profile writers, artists, scholars, and public figures across literature, journalism, film, and politics. Highlights include Thomas Merton, the
Catholic monk and writer honored by a peace award and studied by an international society; W. Somerset Maugham, whose namesake prize supports young writers’
travel; and notable authors such as Mencken, Orwell, Updike, Malamud, Chatwin, and Gass. Other topics include literary awards, bibliographies, influential
editors, Camb
</item 7>

<item 8>
Stephen R. Donaldson
Stephen Reeder Donaldson (born May 13, 1947) is an American fantasy, science fiction and mystery novelist, most famous for "The Chronicles of Thomas Covenant",
his ten-novel fantasy series.

 His work is characterized by psychological complexity, conceptual abstractness, moral bleakness, and the use of an arcane vocabulary, and has attracted
critical praise for its "imagination, vivid characterizations, and fast pace".

 He earned his bachelor's degree from The College of Wooster and a Master's degree from Kent State University.

 He currently resides in New Mexico.
</item 8>

<item 9>
Tom Clancy's Net Force Explorers: Gameprey
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 9>

<item 10>
Tom Clancy's Net Force Explorers: Duel Identity
Tom Clancy's Net Force Explorers or Net Force Explorers is a series of young adult novels created by Tom Clancy and Steve Pieczenik as a spin-off of the
military fiction series Tom Clancy's Net Force.
</item 10>
=== END ITEM id=5e84b951c56ae38b ===

=== ITEM id=262924de0000da7d ===
Gold sentences and facts:
Sentence 0: A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.
  Fact 0.0: A Summer in the Cage is a documentary film.
  Fact 0.1: A Summer in the Cage was released in 2007.
  Fact 0.2: A Summer in the Cage is about a man's experiences with bipolar disorder.
Sentence 1: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.
  Fact 1.0: American Hardcore: The History of American Punk Rock 1980-1986 is a documentary.
  Fact 1.1: American Hardcore: The History of American Punk Rock 1980-1986 was directed by Paul Rachman.
  Fact 1.2: American Hardcore: The History of American Punk Rock 1980-1986 was written by Steven Blush.

Retrieved items:
<item 1>
Hardcore (1979 film)
Hardcore is a 1979 American crime drama film written and directed by Paul Schrader and starring George C. Scott, Peter Boyle and Season Hubley.

 The story concerns a father searching for his daughter, who has vanished only to appear in a pornographic film.

 Writer-director Schrader had previously written the screenplay for Martin Scorsese's "Taxi Driver", and both films share a theme of exploring an unseen
subculture.
</item 1>

<item 2>
Deadfall (1993 film)
Deadfall is a 1993 crime drama film directed by Christopher Coppola.

 Coppola co-wrote the script with Nick Vallelonga.

 The film stars Michael Biehn, Nicolas Cage, Charlie Sheen, James Coburn, and Peter Fonda.

 It is also the prime influence on the song 'Deadfall', written by the American hardcore punk band Snot.

 A prequel/sequel, "Arsenal", starring Nicolas Cage as his character Eddie King, was released in 2017.
</item 2>

<item 3>
Combat Zone Wrestling
Combat Zone Wrestling (CZW) is an American independent wrestling promotion.

 In 1999, John Zandig and five of his students, Ric Blade, T.C.K, Lobo, Nick Gage, and Justice Pain (along with trainer Jon Dahmer) began to run professional
wrestling shows in New Jersey and Delaware, showcasing a brand of hardcore wrestling dubbed as "ultraviolence".

 Ladders, tables, steel folding chairs, thumbtacks, barbed wire, weed whackers, light tubes, panes of glass, and fire are all common elements of "ultraviolent
wrestling" in CZW.

 The company filled a niche for hardcore wrestling fans that had been left open by the folding of Extreme Championship Wrestling (ECW).

 CZW established themselves as the leading American hardcore wrestling promotion at the ECW Arena with their "Cage of Death 3" show in 2001, the year ECW
folded.
</item 3>

<item 4>
Film genre
A film genre is a motion picture category based on similarities in either the narrative elements or the emotional response to the film (namely, serious, comic,
etc.).

 Most theories of film genre are borrowed from literary genre criticism.

 The basic genres include fiction and documentary, from which subgenres have emerged, such as docufiction and docudrama.

 Other subgenres include the courtroom and trial-focused drama known as the legal drama.

 Types of fiction which may seem unrelated can also be combined to form hybrid subgenres, such as the melding of horror and comedy in the "Evil Dead" films.

 Other popular combinations are the romantic comedy and the action comedy film.
</item 4>

<item 5>
American Pie Presents: Band Camp
American Pie Presents: Band Camp (also known as American Pie: Band Camp) is a 2005 direct-to-DVD sex comedy film.

 It is the first installment in the "American Pie Presents" series and the fourth installment in the "American Pie" franchise.

 It was directed by Steve Rash and written by Brad Riddell.

 The film centers on Matt Stifler, younger brother of Steve Stifler.

 In this film, Matt is sent to the infamous band camp, where he is compelled to change his ways.
</item 5>

<item 6>
Hellfest (American music festival)
Hellfest was an American hardcore punk festival.

 Named for the intense summer heat at the show, the festival originally began featuring hardcore bands before crossing into several genres later on.

 Hellfest was known for its fan-friendly atmosphere, (including a lack of barriers, providing easy access to the stage for stage diving) and for its high
social-awareness, focusing on such causes as animal rights, with many attendees following the straight edge and/or vegan lifestyle.

 It continued as an annual event until it was cancelled in 2005.

 After its cancellation less than thirty-six hours before doors were scheduled to open on the festival's ninth year, some fans, unhappy with the inadequate
processing of refunds, filed small claims court lawsuits against promoter Keith Allen.
</item 6>

<item 7>
A Summer in the Cage
A Summer in the Cage is a 2007 documentary film about a man's experiences with bipolar disorder.

 The film follows the filmmaker's friend Sam and features an interview with mental health scholar Kay Redfield Jamison.

 It was directed by Benjamin Selkow.

 The documentary debuted on the Sundance Channel in 2007.
</item 7>

<item 8>
Punk's Not Dead (2007 film)
Punk's Not Dead is a 2007 documentary film directed by Susan Dynner, an American hardcore punk fan.

 The film claims to infiltrate American clubs, malls, recording studios, etc. where it sets out to claim hardcore punk and pop punk music is "thriving" from an
American perspective.

 Its content features performances largely from 1980s hardcore bands and MTV skate punk and pop punk/rock acts.

 It also includes various interviews and behind-the-scenes footage with the bands, labels and fans.
</item 8>

<item 9>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 9>

<item 10>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 10>
=== END ITEM id=262924de0000da7d ===
