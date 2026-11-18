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

=== ITEM id=39897f0a086bcf61 ===
Gold sentences and facts:
Sentence 0: Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.
  Fact 0.0: Tender Fictions is a 1996 autobiographical documentary film.
  Fact 0.1: Tender Fictions was directed by American experimental filmmaker Barbara Hammer.
Sentence 1: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
  Fact 1.0: Bougainville – Our Island Our Fight is a 1998 Australian documentary film.
Sentence 2:  It was produced and directed by Wayne Coles-Janess.
  Fact 2.0: Bougainville – Our Island Our Fight was produced by Wayne Coles-Janess.
  Fact 2.1: Bougainville – Our Island Our Fight was directed by Wayne Coles-Janess.

Retrieved items:
<item 1>
Wayne Coles-Janess
Wayne Coles-Janess is an Australian producer, writer and director of drama and documentary film and TV programs.

 Based in Melbourne, Australia, he has produced documentaries about frontier places in the country.

 He has also made some documentaries in several international locations, including during times of war.
</item 1>

<item 2>
Beyond This Place (2010 film)
Beyond This Place (2010) is a documentary film directed by Kaleo La Belle, screened at a number of film festivals.

 In it, La Belle reunites with his biological father Cloud Rock La Belle, a charismatic figure who was largely absent from his life.

 The film deals, in a personal and intimate way, with issues of parenting, issues of freedom versus responsibility, and with the aging of the 60's generation.
</item 2>

<item 3>
Waiting for Happiness
The film presents typical Mauritanian moments of beauty, struggle, alienation, and humor, which are experienced by groups socially divided from each other, such
as Bidhan women drinking tea and gossiping, West African migrants passing through Mauritania to get to Europe (and finding an unsuccessful comrade washed
ashore).

 The young protagonist who has returned interacts with all of these groups as an outsider, as he struggles to remember even his own Hassaniya Arabic dialect,
but prefers instead French.

 Many of the themes and characters presage Sissako's 2014 film Timbuktu, and both explore liminal Sahel identities authentically situated in everyday life.

 Waiting for Happiness premiered at the 2002 Cannes Film Festival in the Un Certain Regard section.
</item 3>

<item 4>
Bougainville – Our Island Our Fight
Bougainville – Our Island Our Fight is a 1998 Australian documentary film.

 It was produced and directed by Wayne Coles-Janess.
</item 4>

<item 5>
In the Shadow of the Palms
In the Shadow of the Palms (2005) is a documentary produced and directed by the Australian filmmaker Wayne Coles-Janess.

 He filmed it in Iraq prior to, during and after the fall of Saddam Hussein after the United States invasion of 2003.

 An Arab-language film, it documents the changes in Iraqi society and the lives of ordinary Iraqis by focusing on a cross-section of individuals.
</item 5>

<item 6>
The Coconut Revolution
The Coconut Revolution is a 2001 multi-award winning documentary film about the struggle of the indigenous peoples of Bougainville Island during the
Bougainville Civil War.

 The movement is described as the "world's first successful eco-revolution" and has drawn parallels with the conflict depicted in the 2009 film, "Avatar".
</item 6>

<item 7>
Flight of the Red Balloon
Flight of the Red Balloon (French: Le voyage du ballon rouge ) is a 2007 French-Taiwanese film directed by Hou Hsiao-hsien.

 It is the first part in a new series of films produced by Musée d'Orsay, and tells the story of a French family as seen through the eyes of a Chinese student.

 The film was shot in August and September 2006 on location in Paris.

 This is Hou Hsiao-hsien's first non-Asian film.

 It references the classic 1956 French short "The Red Balloon" directed by Albert Lamorisse.
</item 7>

<item 8>
The Boys in Company C
The Boys in Company C, directed by Sidney J. Furie, starring Stan Shaw, Andrew Stevens, Craig Wasson, James Canning, and Michael Lembeck, is a 1978 film about
United States Marine Corps recruits preparing for duty, and their subsequent combat in the Vietnam War.

 It was among the first Vietnam War films to appear after the Vietnam Era, and was also the first role for R. Lee Ermey of "Full Metal Jacket" fame.

 "The Boys in Company C" is the first in Furie's Vietnam War motion picture trilogy, followed by 2001's "Under Heavy Fire" and 2006's "The Veteran".
</item 8>

<item 9>
From the Sea to the Land Beyond
From The Sea to the Land Beyond: Britain's Coast on Film is a documentary feature film directed by Penny Woolcock, with an original soundtrack by British
indie-rock band British Sea Power.

 The project was originally produced by Crossover and Sheffield Doc/Fest as part of The Space project from the BBC and the Arts Council England.

 The film was edited by Alex Fry.
</item 9>

<item 10>
Tender Fictions
Tender Fictions is a 1996 autobiographical documentary film directed by American experimental filmmaker Barbara Hammer.

 It is the second of a trilogy of documentary films that includes "Nitrate Kisses" and "History Lessons".

 Together, the three films are sometimes known as the "History trilogy".

 "Tender Fictions" details Hammer's life and her attempts to "construct" a self.

 The film was nominated for a prize at the 1996 Sundance Film Festival.
</item 10>
=== END ITEM id=39897f0a086bcf61 ===

=== ITEM id=c187e7fe7862b5bd ===
Gold sentences and facts:
Sentence 0: Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story writer.
  Fact 0.0: Doris May Lessing, CH, was born Doris May Tayler (née Tayler).
  Fact 0.1: Doris May Lessing was born on 22 October 1919.
  Fact 0.2: Doris May Lessing died on 17 November 2013.
  Fact 0.3: Doris May Lessing was a British novelist.
  Fact 0.4: Doris May Lessing was a British poet.
  Fact 0.5: Doris May Lessing was a British playwright.
  Fact 0.6: Doris May Lessing was a British librettist.
  Fact 0.7: Doris May Lessing was a British biographer.
  Fact 0.8: Doris May Lessing was a British short story writer.
Sentence 1: Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
  Fact 1.0: Philip Kindred Dick was born on December 16, 1928.
  Fact 1.1: Philip Kindred Dick died on March 2, 1982.
  Fact 1.2: Philip Kindred Dick was an American writer.
  Fact 1.3: Philip Kindred Dick is notable for publishing works of science fiction.

Retrieved items:
<item 1>
Adjustment Team
"Adjustment Team" is a science fiction short story by American writer Philip K. Dick.

 It was first published in "Orbit Science Fiction" (September–October 1954, No. 4) with illustration by Faragasso.

 It was later reprinted in "The Sands of Mars and Other Stories" (Australian) in 1958, "The Book of Philip K. Dick" in 1973, "The Turning Wheel and Other
Stories" (United Kingdom) in 1977, "The Collected Stories of Philip K. Dick" in 1987 (Underwood–Miller), 1988 (Gollancz, United Kingdom), 1990 (Citadel
Twilight, United States), "Selected Stories of Philip K. Dick" in 2002 and in "The Early Work of Philip K. Dick, Volume One: The Variable Man & Other Stories"
in 2009.
</item 1>

<item 2>
Penelope Fitzgerald
Penelope Fitzgerald (17 December 1916 – 28 April 2000) was an English Booker Prize-winning novelist, poet, essayist and biographer.

 In 2008, "The Times" included her in a list of "the 50 greatest British writers since 1945".

 In 2012, "The Observer" named her final novel, "The Blue Flower" one of "the ten best historical novels".
</item 2>

<item 3>
Doris Lessing
Doris May Lessing, CH ("née" Tayler; 22 October 1919  – 17 November 2013) was a British novelist, poet, playwright, librettist, biographer and short story
writer.

 Her novels include "The Grass Is Singing" (1950), the sequence of five novels collectively called "Children of Violence" (1952–69), "The Golden Notebook"
(1962), "The Good Terrorist" (1985), and five novels collectively known as "Canopus in Argos: Archives" (1979–1983).
</item 3>

<item 4>
Philip K. Dick
Philip Kindred Dick (December 16, 1928 – March 2, 1982) was an American writer notable for publishing works of science fiction.
</item 4>

<item 5>
Philip K. Dick Award
The Philip K. Dick Award is a science fiction award given annually at Norwescon sponsored by the Philadelphia Science Fiction Society and (since 2005) supported
by the Philip K. Dick Trust, and named after science fiction and fantasy writer Philip K. Dick.

 It has been awarded since 1983, the year after Dick's death.

 Works that have received the award are identified on their covers as "Best Original SF Paperback".

 They are awarded to the best original paperback published each year in the US.
</item 5>

<item 6>
The Philip K. Dick Reader
The Philip K. Dick Reader is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Citadel Twilight in 1997.

 Many of the stories had originally appeared in the magazines "If", "Science Fiction Adventures", "Science Fiction Stories", "Orbit", "Fantasy and Science
Fiction", "Imagination", "Future", "Galaxy Science Fiction", "Beyond Fantasy Fiction", "Satellite", "Imaginative Tales", "Fantastic Universe" and "Space Science
Fiction".

 It is identical in content and order to the edition of volume 3 of the Collected Stories of Philip K. Dick produced by the same publisher apart from the
substitution of three stories in positions 21-23 of 24 and the omission of the end notes in the Collected Stories edition.

 At press time, stories 21 and 24 had already been made into successful movie adaptations and stories 22 and 23 had been optioned.
</item 6>

<item 7>
The Good Terrorist
The Good Terrorist is a 1985 political novel written by the British novelist Doris Lessing.

 It was first published in September that year by Jonathan Cape in the United Kingdom and by Alfred A. Knopf in the United States.

 The book's protagonist is the naïve drifter Alice, who squats with a group of radicals in London and is drawn into their terrorist activities.
</item 7>

<item 8>
Jenny Diski
Jenny Diski FRSL (née Simmonds; 8 July 1947 – 28 April 2016) was an English writer.

 She had a troubled childhood, but was rescued by the older novelist Doris Lessing; she lived in Lessing's house for four years.

 Diski was educated at University College London, and worked as a teacher during the 1970s and early 1980s.
</item 8>

<item 9>
The Days of Perky Pat (collection)
The Days of Perky Pat is a collection of science fiction stories by American writer Philip K. Dick.

 It was first published by Gollancz in 1990 and reprints Volume IV of "The Collected Stories of Philip K. Dick".

 It had not previously been published as a stand-alone volume.

 The stories had originally appeared in the magazines "Galaxy Science Fiction", "Science Fiction Stories", "If", "Fantastic Universe", "Fantasy and Science
Fiction", "Fantastic", "Worlds of Tomorrow", "Escapade" and "Amazing Stories".
</item 9>

<item 10>
Herbert Lottman
Herbert Lottman (August 16, 1927, Brooklyn - August 27, 2014, Paris) was an American author who specialized in writing biographies on French subjects.

 An influential biographer, he published 17 biographies, 15 of which were related to French culture, commerce, or politics; including works on Albert Camus,
Colette, Gustave Flaubert, Henri Philippe Pétain, Jules Verne, and the Rothschild banking family of France.

 He wrote that, just before dying, Albert Camus was pledged to marry.

 Camus's estate tried to block his book, partly because of this controversial statement.
</item 10>
=== END ITEM id=c187e7fe7862b5bd ===

=== ITEM id=e883efeeca28959b ===
Gold sentences and facts:
Sentence 0: Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.
  Fact 0.0: Francis Keogh Gleason was born on April 14, 1906.
  Fact 0.1: Francis Keogh Gleason died on December 18, 1982.
  Fact 0.2: Francis Keogh Gleason was a resident set decorator at MGM studios.
  Fact 0.3: Francis Keogh Gleason was a resident set decorator at MGM studios for over 40 years.
Sentence 1:  In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi" in 1958) and was nominated an additional 3 times.
  Fact 1.0: Francis Keogh Gleason won 4 Academy Awards during his time at MGM.
  Fact 1.1: Francis Keogh Gleason won an Academy Award for 'An American in Paris' in 1951.
  Fact 1.2: Francis Keogh Gleason won an Academy Award for 'The Bad and the Beautiful' in 1952.
  Fact 1.3: Francis Keogh Gleason won an Academy Award for 'Somebody Up There Likes Me' in 1956.
  Fact 1.4: Francis Keogh Gleason won an Academy Award for 'Gigi' in 1958.
  Fact 1.5: Francis Keogh Gleason was nominated for an Academy Award an additional 3 times.
Sentence 2: Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.
  Fact 2.0: Somebody Up There Likes Me is a 1956 American drama film.
  Fact 2.1: Somebody Up There Likes Me is based on the life of middleweight boxing legend Rocky Graziano.

Retrieved items:
<item 1>
Rocky Graziano
Thomas Rocco Barbella (January 1, 1919 – May 22, 1990), better known as Rocky Graziano, was an American professional boxer who held the World Middleweight
title.

 Graziano is considered one of the greatest knockout artists in boxing history, often displaying the capacity to take his opponent out with a single punch.

 He was ranked 23rd on "The Ring" magazine list of the greatest punchers of all time.

 He fought many of the best middleweights of the era including Sugar Ray Robinson.

 His turbulent and violent life story was the basis of the 1956 Oscar-winning drama film, "Somebody Up There Likes Me", based on his 1955 autobiography of the
same title.
</item 1>

<item 2>
Somebody Up There Likes Me (1956 film)
Somebody Up There Likes Me is a 1956 American drama film based on the life of middleweight boxing legend Rocky Graziano.

 Joseph Ruttenberg was awarded a 1956 Oscar in the category of Best Cinematography (Black and White).

 The film also won the Oscar for Best Art Direction (Cedric Gibbons, Malcolm F. Brown, Edwin B. Willis, F. Keogh Gleason).

 It was directed by Robert Wise.
</item 2>

<item 3>
Brian Ralston
Brian Ralston (born April 12, 1974) is a composer and musician living in Los Angeles.

 Ralston is a graduate of the University of Arizona and the USC Thornton School of Music Scoring for Motion Pictures and Television program.

 Brian's latest film is the 2017 drama Rose starring Cybill Shepherd, James Brolin and Pam Grier.

 In 2012 he scored the 20th Century Fox inspirational sports film Crooked Arrows, starring Brandon Routh, directed by Steve Rash.

 He has also composed music for the television series "Angel" (Season 4) and scores to the theatrical motion pictures "9/Tenths", directed by Bob Degus
("Pleasantville") starring Gabrielle Anwar, Henry Ian Cusick and Dave Ortiz, the Magnolia Pictures teen heist movie Graduation directed by Mike Mayer and
starring Adam Arkin, Shannon Lucio, Chris Marquette, Riley Smith and Chris Lowell, and the dramatic feature Don't Fade Away directed by Luke Kasdan, starring
Beau Bridges, Mischa Barton and Ryan Kwanten.
</item 3>

<item 4>
Rocky V
Rocky V is a 1990 American sports drama film.

 It is the fifth film in the "Rocky" series, written by and starring Sylvester Stallone, and co-starring Talia Shire, Stallone's real-life son Sage, and
real-life boxer Tommy Morrison, with Morrison in the role of Tommy Gunn, a talented yet raw boxer.

 Sage played Robert Balboa, whose relationship with his famous father is explored.

 After Stallone directed the second through fourth films in the series, "Rocky V" saw the return of John G. Avildsen, whose direction of "Rocky" won him an
Academy Award for Best Directing.
</item 4>

<item 5>
Richard Farnsworth
Richard W. Farnsworth (September 1, 1920 – October 6, 2000) was an American actor and stuntman.

 He is best known for his performances in "The Grey Fox" (1982), for which he received a Golden Globe Award nomination, "Anne of Green Gables" (1985), "Misery"
(1990), and "The Straight Story" (1999), for which he received a nomination for the Academy Award for Best Actor.

 His sudden death followed a long struggle with prostate cancer.
</item 5>

<item 6>
The Life of Emile Zola
The Life of Emile Zola is a 1937 American biographical film about French author Émile Zola, played by Paul Muni and directed by William Dieterle.

 It has the distinction of being the second biographical film to win the Oscar for Best Picture.

 It premiered at the Los Angeles Carthay Circle Theatre to great success both critically and financially.

 Contemporary reviews cited it the best biographical film made up to that time.

 In 2000, it was selected for preservation in the United States National Film Registry by the Library of Congress as being "culturally, historically, or
aesthetically significant".
</item 6>

<item 7>
The Man with the Golden Arm
The Man with the Golden Arm is a 1955 American drama film with elements of film noir, based on the novel of the same name by Nelson Algren, which tells the
story of a drug addict who gets clean while in prison, but struggles to stay that way in the outside world.

 Although the addictive drug is never identified in the film, according to the American Film Institute "most contemporary and modern sources assume that it is
heroin", in contrast to Algren's book which named the drug as morphine.

 The film stars Frank Sinatra, Eleanor Parker, Kim Novak, Arnold Stang and Darren McGavin.

 It was adapted for the screen by Walter Newman, Lewis Meltzer and Ben Hecht (uncredited), and directed by Otto Preminger.

 The film's initial release sparked controversy due to its serious, in-depth treatment of the then-taboo subject of drug addiction.
</item 7>

<item 8>
Paul Schrader
Paul Joseph Schrader (born July 22, 1946) is an American screenwriter, film director, and film critic.

 Schrader wrote or co-wrote screenplays for four Martin Scorsese films: "Taxi Driver" (1976), "Raging Bull" (1980), "The Last Temptation of Christ" (1988), and
"Bringing Out the Dead" (1999).

 Schrader has also directed 18 feature films, including his directing debut crime drama, "Blue Collar" (co-written with his brother, Leonard Schrader), the
crime drama "Hardcore" (a loosely autobiographical film also written by Schrader), his 1982 remake of the horror classic "Cat People", the crime drama "American
Gigolo" (1980), the biographical drama "" (1985), the cult film "Light Sleeper" (1992), the drama "Affliction" (1997), the biographical film "Auto Focus"
(2002), and the erotic dramatic thriller "The Canyons" (2013).
</item 8>

<item 9>
Academy Award for Best Film Editing
The Academy Award for Best Film Editing is one of the annual awards of the Academy of Motion Picture Arts and Sciences (AMPAS).

 Nominations for this award are closely correlated with the Academy Award for Best Picture.

 For 33 consecutive years, 1981 to 2013, every Best Picture winner had also been nominated for the Film Editing Oscar, and about two thirds of the Best Picture
winners have also won for Film Editing.

 Only the principal, "above the line" editor(s) as listed in the film's credits are named on the award; additional editors, supervising editors, etc. are not
currently eligible.

 The nominations for this Academy Award are determined by a ballot of the voting members of the Editing Branch of the Academy; there were 220 members of the
Editing Branch in 2012.

 The members may vote for up to five of the eligible films in the order of their preference; the five films with the largest vote totals are selected as
nominees.

 The Academy Award itself is selected from the nominated films by a subsequent ballot of all active and life members of the Academy.
</item 9>

<item 10>
F. Keogh Gleason
Francis Keogh Gleason (April 14, 1906 – December 18, 1982) was a resident set decorator at MGM studios for over 40 years.

 In that time he won 4 Academy Awards (for "An American in Paris" in 1951, "The Bad and the Beautiful" in 1952, "Somebody Up There Likes Me" in 1956 and "Gigi"
in 1958) and was nominated an additional 3 times.

 Gleason graduated from the Minnesota School of Fine Arts (now known as the Minneapolis College of Art and Design) in 1927.
</item 10>
=== END ITEM id=e883efeeca28959b ===

=== ITEM id=8bbc61118b85599e ===
Gold sentences and facts:
Sentence 0: Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York, United States.
  Fact 0.0: Albany International Airport has the IATA code ALB.
  Fact 0.1: Albany International Airport has the ICAO code KALB.
  Fact 0.2: Albany International Airport has the FAA LID ALB.
  Fact 0.3: Albany International Airport is a public airport.
  Fact 0.4: Albany International Airport is seven miles (11 km) northwest of Albany.
  Fact 0.5: Albany International Airport is in Albany County, New York, United States.
Sentence 1: Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown Madison, the capital of Wisconsin.
  Fact 1.0: Dane County Regional Airport (DCRA) has the IATA code MSN.
  Fact 1.1: Dane County Regional Airport (DCRA) has the ICAO code KMSN.
  Fact 1.2: Dane County Regional Airport (DCRA) has the FAA LID MSN.
  Fact 1.3: Dane County Regional Airport is also known as Truax Field.
  Fact 1.4: Dane County Regional Airport is a civil-military airport.
  Fact 1.5: Dane County Regional Airport is located six miles northeast of downtown Madison.
  Fact 1.6: Madison is the capital of Wisconsin.

Retrieved items:
<item 1>
Dane County Regional Airport
Dane County Regional Airport (DCRA) (IATA: MSN, ICAO: KMSN, FAA LID: MSN) (Truax Field) is a civil-military airport located six miles northeast of downtown
Madison, the capital of Wisconsin.

 In the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, it is categorized as a small-hub primary commercial
service facility.
</item 1>

<item 2>
Truax Field Air National Guard Base
Truax Field Air National Guard Base (IATA: MSN, ICAO: KMSN, FAA LID: MSN) , also known as Truax Field, is a military facility located at Dane County Regional
Airport.

 It is located five miles (8 km) northeast of the center of Madison, a city in Dane County, Wisconsin, United States.
</item 2>

<item 3>
Grand Forks Municipal Airport
Grand Forks Municipal Airport is a former airport, located within current city limits approximately 3 mi west-northwest of central Grand Forks, North Dakota.

 It was closed shortly after World War II and is now redeveloped as part of the urban area of Grand Forks.
</item 3>

<item 4>
Lafayette Regional Airport
Lafayette Regional Airport (IATA: LFT, ICAO: KLFT, FAA LID: LFT) is a public airport two miles (4 km) southeast of Lafayette, in Lafayette Parish, Louisiana,
United States.

 It is independently owned and operated.
</item 4>

<item 5>
Wilmington Airport (Delaware)
Wilmington Airport (IATA: ILG) (also known as Wilmington/Philadelphia Regional Airport, New Castle County Airport, or New Castle Airport) is an airport located
in unincorporated New Castle County, Delaware near Wilmington, Delaware.

 Owned by the Delaware River and Bay Authority, it is five miles (8 km) south of Wilmington and about 30 miles (50 km) from Philadelphia.

 It is included in the Federal Aviation Administration (FAA) National Plan of Integrated Airport Systems for 2017–2021, in which it is categorized as a non-hub
primary commercial service facility.
</item 5>

<item 6>
Albany International Airport
Albany International Airport (IATA: ALB, ICAO: KALB, FAA LID: ALB) is a public airport seven miles (11 km) northwest of Albany, in Albany County, New York,
United States.

 It is owned by the Albany County Airport Authority.
</item 6>

<item 7>
Brainerd Lakes Regional Airport
Brainerd Lakes Regional Airport (IATA: BRD, ICAO: KBRD, FAA LID: BRD) is a public use airport located three nautical miles (6 km) northeast of the central
business district of Brainerd, a city in Crow Wing County, Minnesota, United States.

 The airport is owned by the city and county.

 It is mostly used for general aviation but is also served by one commercial airline.
</item 7>

<item 8>
Lea County Regional Airport
Lea County Regional Airport (IATA: HOB, ICAO: KHOB) (Lea County-Hobbs Airport) is four miles (6.4 km) west of Hobbs, in Lea County, New Mexico.

 The airport covers 898 acre and has three runways.

 It is an FAA certified commercial airport served by United Airlines' affiliate with daily regional flights.

 Lea County Regional Airport is the largest of the three airports owned and operated by Lea County Government.

 Lea County also owns and operated two general aviation airports in Lovington and Jal, New Mexico.
</item 8>

<item 9>
Raleigh Executive Jetport
Raleigh Exec: The Raleigh Executive Jetport @ Sanford-Lee County or Raleigh Exec Jetport at Sanford-Lee CountyFAA Airport Master Record for TTA (Form 5010 )
(ICAO: KTTA, FAA LID: TTA) is a public use airport located seven nautical miles (8 mi, 13 km) northeast of the central business district of Sanford, a city in
Lee County, North Carolina, United States.

 It is owned by the Sanford-Lee County Regional Airport Authority and was previously known as Sanford-Lee County Regional Airport.

 This airport is included in the National Plan of Integrated Airport Systems for 2011–2015, which categorized it as a "reliever airport" for Raleigh-Durham
International Airport.
</item 9>

<item 10>
Air Wisconsin
Air Wisconsin Airlines is a regional airline based at Appleton International Airport in the town of Greenville, Wisconsin, United States, near Appleton.

 Air Wisconsin is the largest privately held regional airline in the United States.

 It currently operates regional jet flights as American Eagle under contract to American Airlines via a code sharing agreement, serving cities in the U.S. and
Canada with hubs at Boston Logan International Airport (BOS), Charlotte Douglas International Airport (CLT), Philadelphia International Airport (PHL) and Ronald
Reagan Washington National Airport (DCA) .

 Air Wisconsin previously operated United Express service on behalf of United Airlines followed by US Airways Express service on behalf of US Airways prior to
becoming an American Eagle regional air carrier.

 The company has announced it will once again partner with United Airlines as a United Express code sharing air carrier with primary hubs to be located at
Chicago O'Hare International Airport (ORD) and Washington Dulles International Airport (IAD) beginning in September 2017 and will be "exclusively operating" as
United Express by March 2018.
</item 10>
=== END ITEM id=8bbc61118b85599e ===

=== ITEM id=920d9203c54c8125 ===
Gold sentences and facts:
Sentence 0: Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.
  Fact 0.0: Samantha Cristoforetti was born 26 April 1977 in Milan.
  Fact 0.1: Samantha Cristoforetti is an Italian European Space Agency astronaut.
  Fact 0.2: Samantha Cristoforetti is an Italian Air Force pilot.
  Fact 0.3: Samantha Cristoforetti is an Italian engineer.
Sentence 1: Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force), and a former CNES spationaut.
  Fact 1.0: Jean-Loup Jacques Marie Chrétien was born 20 August 1938.
  Fact 1.1: Jean-Loup Jacques Marie Chrétien is French.
  Fact 1.2: Jean-Loup Jacques Marie Chrétien is retired.
  Fact 1.3: Jean-Loup Jacques Marie Chrétien holds the rank of Général de Brigade (brigadier general).
  Fact 1.4: Jean-Loup Jacques Marie Chrétien served as a general in the Armée de l'Air (French air force).
  Fact 1.5: Jean-Loup Jacques Marie Chrétien is a former CNES spationaut.

Retrieved items:
<item 1>
Soyuz T-6
Soyuz T-6 was a manned spaceflight to Earth orbit to the Salyut 7 space station in 1982.

 Along with two Soviet cosmonauts, the crew included a Frenchman, Jean-Loup Chrétien.
</item 1>

<item 2>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 2>

<item 3>
Jean-Loup Puget
Jean-Loup Puget (born 7 March 1947) is a French astrophysicist.

 His current research interests lie in the Cosmic Microwave Background.

 Jean-Loup Puget and his collaborators reported the first identification of the Cosmic infrared background using COBE data.

 He is also, along with Alain Léger, credited with the origin of the hypothesis that the series of infrared lines observed in numerous astrophysical objects are
caused by emission from polycyclic aromatic hydrocarbons.

 He is currently principal investigator of the HFI module of the Planck space mission.
</item 3>

<item 4>
Jean-Loup Chrétien
Jean-Loup Jacques Marie Chrétien (born 20 August 1938) is a French retired "Général de Brigade" (brigadier general) in the "Armée de l'Air" (French air force),
and a former CNES spationaut.

 He flew on two Franco-Soviet space missions and a NASA Space Shuttle mission.

 Chrétien was the first Frenchman and the first western European in space.
</item 4>

<item 5>
Samantha Cristoforetti
Samantha Cristoforetti (] ; born 26 April 1977 in Milan) is an Italian European Space Agency astronaut, Italian Air Force pilot and engineer.

 She holds the record for the longest uninterrupted spaceflight of a European astronaut (199 days, 16 hours), and until June 2017 held the record for the
longest single space flight by a woman until this was broken by Peggy Whitson.

 She is also the first Italian woman in space.

 Samantha Cristoforetti is also known as the first person who brewed an espresso coffee in space.
</item 5>

<item 6>
Philippe Perrin
Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.
</item 6>

<item 7>
Jean-François Clervoy
Jean-François André Clervoy (born 19 November 1958) is a French engineer and a CNES and ESA astronaut.

 He is a veteran of three NASA Space Shuttle missions.
</item 7>

<item 8>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 8>

<item 9>
John Christiansen
John “Chris” Christiansen (May 1, 1923 － September 12, 1998) was the chief military test pilot for Lockheed California Company for over 30 years.

 He might be most known for having performed Lockheed S-3 Viking's maiden flight on January 21, 1972.

 His assignments also included the initial test flights of Lockheed P-3 Orion.

 Christiansen was born in Oslo, Norway in 1923 and became an American citizen in 1939.

 He later served in the U.S. Navy during World War II and the Korean Conflict.

 He began experimental flying for Lockheed Martin in 1953, and worked there until his retirement in 1984.

 He was a fellow at Society of Experimental Test Pilots.
</item 9>

<item 10>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 10>
=== END ITEM id=920d9203c54c8125 ===

=== ITEM id=5827a10773668cd5 ===
Gold sentences and facts:
Sentence 0: Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.
  Fact 0.0: Marc Predka was born April 21, 1975.
  Fact 0.1: Marc Predka is better known by his stage name Tha Trademarc.
  Fact 0.2: Marc Predka is an American hip hop artist.
Sentence 1:  He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance theme "The Time Is Now".
  Fact 1.0: Marc Predka first came to prominence when he collaborated with his first cousin, John Cena.
  Fact 1.1: John Cena is a WWE wrestler.
  Fact 1.2: Marc Predka and John Cena collaborated on the 2005 album 'You Can't See Me'.
  Fact 1.3: The album 'You Can't See Me' includes Cena's entrance theme 'The Time Is Now'.
Sentence 2: "The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.
  Fact 2.0: 'The Time Is Now' is a song by John Cena.
  Fact 2.1: John Cena is a professional wrestler.
  Fact 2.2: John Cena is an actor.
  Fact 2.3: John Cena is a hip hop musician.

Retrieved items:
<item 1>
Rebuild (song)
"Rebuild" is a song written by Jon Foreman of Switchfoot and Matt Thiessen of Relient K.

 The song was written in conjunction with the bands' 2007 Appetite for Construction Tour, featuring members from all three bands on tour, Switchfoot, Relient K,
and Ruth.

 Thiessen sang second vocals, and Dustin Ruth of Ruth played harmonica.
</item 1>

<item 2>
Not Your Fault
"Not Your Fault" is a song by American alternative rock band Awolnation.

 It was originally written by frontman Aaron Bruno and recorded by the band for their debut studio album "Megalithic Symphony", where it appears as the twelfth
track.

 "Not Your Fault" was released to American modern rock radio on October 17, 2011, and was also released as a single in the United Kingdom and Bosnia on February
26, 2012.

 The single peaked at numbers three and eleven on the US "Billboard" Alternative Songs and Rock Songs charts, respectively.
</item 2>

<item 3>
Nobody Else (CeCe Peniston song)
"Nobody Else" is a 1998 song by CeCe Peniston.

 The composition was to be the lead single of the singer's originally scheduled album release on the Silk Entertainment label, as well as reportedly released on
Steve Hurley's compilation "The Voices of Life, Vol 1."

 The production of her album was cancelled eventually, and releasing of the compilation wouldn't happen either.

 However, the singer released another two singles on the label, "He Loves Me 2" the following year and "My Boo" in 2000.
</item 3>

<item 4>
I Would Like to See You Again
I Would Like to See You Again is the 57th album by American country singer Johnny Cash, released on Columbia Records in 1978.

 The title track peaked at #12 on the singles chart, while "There Ain't No Good Chain Gang" reached #2; the album itself peaked at #23.

 The album features a pair of duets with Waylon Jennings, one of which was the "There Ain't No Good Chain Gang" single; it was one of Cash's first
collaborations with Jennings, and the two recorded songs together throughout the 1980s, including a separate album entitled "Heroes".

 Cash and Jennings would also work together as The Highwaymen with Willie Nelson and Kris Kristofferson.
</item 4>

<item 5>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 5>

<item 6>
Greg Davis (musician)
Greg Davis is an American electronic musician who has recorded albums drawing from a wide variety of sources, including guitar, field recording, various
world/ethnic/traditional instruments, percussion, and voice, all processed through digital manipulation.

 After years of experience in hip-hop groups, jazz combos, free improvisation, and experimental composition, Davis moved toward computer-based music in 1997.

 At Depaul University in Chicago, he studied classical and jazz guitar alongside composition and jazz studies.

 In 1997, he started his own label, Autumn Records, to release his own music and the music of others.

 Davis moved to Boston in 1999 to pursue a master's degree in composition (which he received in June 2001) while independently immersing himself in the study of
electronic music.

 Since 2001, he has played shows all around the world and released records on a number of different labels.

 He has collaborated with Keith Fullerton Whitman, Ben Vida, Sebastien Roux, Chris Weisman, Steven Hess, Jeph Jerman, Akron/Family, Toby Aronson (as
Harmonizer), and Zach Wallace (as Sun Circle).
</item 6>

<item 7>
Treat (album)
Treat is a split cassette shared between by Dutch punk band The Ex and Scottish ex-pat tour mates Dog Faced Hermans.

 The album was recorded live while the two bands toured Europe together and was released only on cassette in 1990.

 That year the two bands also collaborated on the single "Lied der Steinklopfer" ("Stonestamper's Song") released under the name Ex Faced Hermans, as well as
sharing live sound engineer Gert-Jan, credited as a full member of the Dog Faced Hermans who continued to tour with The Ex for more than a decade.
</item 7>

<item 8>
John Fogerty (album)
John Fogerty is the second solo studio album by former Creedence Clearwater Revival vocalist/guitarist John Fogerty, released in 1975.

 It was released by Asylum Records in the United States and Fantasy Records internationally.

 As with the Creedence Clearwater Revival records, the album consists of a mix of originals and cover songs.

 Although the album is eponymously titled, Fogerty himself refers to it as "Old Shep"; Shep was the name of his dog, who appears on the cover with him.
</item 8>

<item 9>
Tha Trademarc
Marc Predka (born April 21, 1975) better known by his stage name Tha Trademarc, is an American hip hop artist.

 He first came to prominence when he and his first cousin, WWE wrestler John Cena, collaborated on the 2005 album "You Can't See Me", including Cena's entrance
theme "The Time Is Now".

 He also appeared in the music videos of the songs, "Bad Bad Man" and "Right Now" with Cena.

 Prior to the album, Trademarc appeared on Cena's first WWE DVD "Word Life".

 He now works as a classroom specialist for the Peace Corps of America.
</item 9>

<item 10>
The Time Is Now (John Cena song)
"The Time Is Now" is a song by professional wrestler, actor, and hip hop musician John Cena.

 It is the current entrance theme Cena uses as he makes his entrance to the ring.

 Currently wrestling for WWE, Cena recorded the song in 2005 for his debut studio album, "You Can't See Me".
</item 10>
=== END ITEM id=5827a10773668cd5 ===

=== ITEM id=517e20a1f1631e3f ===
Gold sentences and facts:
Sentence 0:  The name comes from the City of Starachowice, where the factory is located.
  Fact 0.0: The name comes from the City of Starachowice.
  Fact 0.1: The factory is located in the City of Starachowice.
Sentence 1:  It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).
  Fact 1.0: The item was sold in various countries for many years.
  Fact 1.1: The item was sold in countries beyond the Eastern Bloc.
  Fact 1.2: The item was used by the Yemen Army.
Sentence 2: The Eastern Bloc was the group of socialist states of Central and Eastern Europe, generally the Soviet Union and the countries of the Warsaw Pact.
  Fact 2.0: The Eastern Bloc was the group of socialist states of Central and Eastern Europe.
  Fact 2.1: The Eastern Bloc generally included the Soviet Union.
  Fact 2.2: The Eastern Bloc generally included the countries of the Warsaw Pact.

Retrieved items:
<item 1>
Optimus S.A.
Optimus S.A. was a Polish company specializing in assembly of PCs, fiscal systems, and providing complex IT solutions for companies.In the 1990s Optimus was one
of the leaders of the Polish computer market.
</item 1>

<item 2>
Soviet Unterzoegersdorf
Soviet Unterzoegersdorf (German: "Sowjet-Unterzögersdorf" ) is a fictitious country created by the art/technology/theory group monochrom.

 It is the "last existing appanage republic of the USSR", located inside the Republic of Austria.

 Unterzögersdorf is partially based on an existing village well-known to Johannes Grenzfurthner, the creator of the concept.

 As a kid and teenager, Grenzfurthner spent a lot of time at his grandparents' farm in the small village of Unterzögersdorf (a cadastral municipality of
Stockerau).

 His grandparents' stories about the Naziism, WWII and the Soviet occupation in allied-occupied Austria (1945-1955) form the inspirational basis of the
long-term project "Soviet Unterzoegersdorf".
</item 2>

<item 3>
Leader Trucks
Leader found their niche in the heavy transport market and built as a rigid 4x4 and 6x6 configuration for tray, tipper and agitator applications, they offered
reliable machines in the industries of earth-moving and off-road construction.

 Leader was notable for being the first manufacturer to fit Caterpillar engines to trucks, to fit automatic transmissions to diesel trucks in assembly, and the
first truck manufacturer in Australia to offer disc brakes.

 Leader boasted its components to be 80 percent Australian built with only the engine, transmission and steering box imported.

 By 1980, Leader had sold its 1000th truck in Australia and had also sold trucks to New Zealand, South Africa, Hong Kong, China and Indonesia.
</item 3>

<item 4>
Kolejka (game)
Kolejka ( ; Polish for "queue" or "line (of people)") is a board game designed by Karol Madaj and launched in February 2011 by Poland's Institute of National
Remembrance.

 It was designed to teach young generations about the hardships under communism (1945–1989), particularly the difficulty of acquiring consumer goods in the
shortage economy.

 The game has been described as inspired by "Monopoly".
</item 4>

<item 5>
FSC Star
Fabryka Samochodów Ciężarowych "Star" was a Polish truck manufacturer.

 The name comes from the City of Starachowice, where the factory is located.

 Their first vehicle was the "Star 20" in 1948.

 The most popular product was the "Star 266".

 The 266 model offered very good quality and powerful engines for a low price.

 It was sold in various countries for many years (not only in the Eastern Bloc; for example it was used by the Yemen Army).

 For many years FSC Star was a state-owned company.

 Star is now owned by MAN AG who eliminated the brand in January 2009.
</item 5>

<item 6>
Kresy
Kresy Wschodnie or Kresy (] , Eastern Borderlands, or Borderlands) was a region of the Second Polish Republic during the interwar period constituting nearly
half of the territory of the state; where the ethnic Poles, being the largest group, were roughly equal in their number to the size of the national minorities
(with notable exceptions).

 Administratively, the territory of Kresy was composed of voivodeships of Lwów, Nowogródek, Polesie, Stanisławów, Tarnopol, Wilno, Wołyń, and the Białystok.

 Today, these territories are divided between Western Ukraine, Western Belarus, and south-eastern Lithuania, with such major cities as Lviv, Vilnius, and Grodno
no longer in Poland.

 In the Second Polish Republic the term "Kresy" roughly equated with the lands beyond the so-called Curzon Line, which was suggested after World War I in
December 1919 by the British Foreign Office as the eastern border of the re-emerging sovereign Republic following the century of partitions.
</item 6>

<item 7>
Daimler AG
Daimler AG (] ) is a German multinational automotive corporation.

 Daimler AG is headquartered in Stuttgart, Baden-Württemberg, Germany.

 As of 2014, Daimler owns or has shares in a number of car, bus, truck and motorcycle brands including Mercedes-Benz, Mercedes-AMG, Smart Automobile, Detroit
Diesel, Freightliner, Western Star, Thomas Built Buses, Setra, BharatBenz, Mitsubishi Fuso, MV Agusta as well as shares in Denza, KAMAZ, Beijing Automotive
Group.

 The luxury Maybach brand was terminated at the end of 2012, but revived in April 2015 as "Mercedes-Maybach" versions of the Mercedes-Benz S-Class and G-Class.

 In 2016 Daimler sold 3.0 million vehicles.

 By unit sales, Daimler is the thirteenth-largest car manufacturer and is the largest truck manufacturer in the world.

 In addition to automobiles, Daimler manufactures buses and provides financial services through its Daimler Financial Services arm.

 The company is a component of the Euro Stoxx 50 stock market index.
</item 7>

<item 8>
J. A. Baczewski
J. A. Baczewski was a Polish company most well known for its fine spirits such as vodka and gin.

 The factory, dating back to the late 18th century, was based in Lwów (Lviv) and until 1939 was one of two most popular Polish export goods.
</item 8>

<item 9>
Shaanxi Automobile Group
Shaanxi Automobile Group Co., Ltd. is a Chinese bus and truck manufacturer headquartered in Xi'an, Shaanxi.

 It employs around 23,000 employees.

 It manufactures heavy duty truck, buses (chassis), medium-size heavy truck and heavy duty axles for trucks.

 It utilizes Magna Steyr and MAN SE technologies.

 Its bus chassis are sold under the Eurostar Bus brand.
</item 9>

<item 10>
Kross SA
Kross is a Polish company within the sports industry that makes bicycles and bicycle frames.

 The company is currently the largest in Poland manufacturing bicycles sold in the Polish market.

 In 2005 the company sold almost 800,000 products, half of which were exported to 35 countries, 30 of which are in Europe.
</item 10>
=== END ITEM id=517e20a1f1631e3f ===

=== ITEM id=81edf131fcc7079e ===
Gold sentences and facts:
Sentence 0: Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.
  Fact 0.0: Lee Dong-hae was born on October 15, 1986.
  Fact 0.1: Lee Dong-hae's name in Hangul is 이동해.
  Fact 0.2: Lee Dong-hae's name in Hanja is 李東海.
  Fact 0.3: Lee Dong-hae is a South Korean singer.
  Fact 0.4: Lee Dong-hae is a South Korean songwriter.
  Fact 0.5: Lee Dong-hae is a South Korean actor.
Sentence 1: Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.
  Fact 1.0: Costadinos Contostavlos was born on 11 June 1987.
  Fact 1.1: Costadinos Contostavlos is better known by his stage name Dappy.
  Fact 1.2: Costadinos Contostavlos is an English singer.
  Fact 1.3: Costadinos Contostavlos is an English songwriter.
  Fact 1.4: Costadinos Contostavlos is an English rapper.
  Fact 1.5: Costadinos Contostavlos is an English actor.

Retrieved items:
<item 1>
List of songs written by Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Donghae & Eunhyuk, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 1>

<item 2>
As One (musical duo)
As One () is a South Korean R&B duo consisting of Korean-American singers Lee Min-young (known as Lee Min) and Chae Da-hee (known as Crystal).

 They are signed to record label Brand New Music.
</item 2>

<item 3>
Dappy
Costadinos Contostavlos (born 11 June 1987), better known by his stage name Dappy, is an English singer, songwriter, rapper, and actor.

 He is best known for being the lead singer of the Camden-based grime trio N-Dubz, with his cousin Tulisa, and Fazer.

 He was known for his slightly unusual dress sense and his love for what he describes as "eye-catching headwear".

 His trademark was wearing a selection of woolly chullos (sometimes referred to as "Dappy hats"), turning up one or both of the ear-flaps of the hat.
</item 3>

<item 4>
Ms Panda and Mr Hedgehog
Panda and Hedgehog () is a 2012 South Korean romantic comedy television series starring Lee Donghae of Super Junior and Yoon Seung-ah.

 Produced by Song Hae-sung's TV production venture Lion Fish, it aired on Channel A from August 18 to October 7, 2012 on Saturdays and Sundays at 21:55 for 16
episodes.
</item 4>

<item 5>
Kim Dong-jun
Kim Dong-jun (hangul: 김동준; hanja: 金桐俊; born February 11, 1992) is a South Korean singer and actor.

 He is a member of the South Korean boyband and its sub-group ZE:A Five and ZE:A J.
</item 5>

<item 6>
Lee Donghae
Lee Dong-hae (hangul: 이동해; hanja: 李東海; born October 15, 1986) is a South Korean singer, songwriter and actor.

 He is a member of the boy band Super Junior, its subgroup Super Junior-M, and Super Junior-D&E, as well as the dance-centered group SM The Performance.

 He is one of the first four Korean artists to appear on Chinese postage stamps.
</item 6>

<item 7>
Danny Shirley
Danny Shirley (born August 12, 1956) is an American country music singer.

 He is best known as the lead singer of the country rock band Confederate Railroad, a role he has held since its formation in 1987.
</item 7>

<item 8>
Lee Won-il
Lee Won-il (; born December 12, 1979) is a South Korean chef and television personality.

 He is currently a cast member in the variety show "Please Take Care of My Refrigerator".

 He is owner and chef of "Dear.

 bread".
</item 8>

<item 9>
Eunhyuk
Lee Hyuk-jae (born April 4, 1986), better known by his stage name Eunhyuk, is a South Korean singer-songwriter and actor.

 He is a member of the South Korean boy group Super Junior and their subgroups, Super Junior-T and Super Junior-H.

 In 2011, he joined Super Junior's Mandopop subgroup, Super Junior-M and the duo Donghae & Eunhyuk; active in China and Japan respectively.
</item 9>

<item 10>
ELDee
Lanre Dabiri (born May 23, 1977), better known by his stage name Eldee, stylized as eLDee, is a former Nigerian-American rapper, singer, and record producer but
now an IT Consultant based in the United_States "Lanre" is a diminutive for the Yoruba name "Olanrewaju" (which translates to ""Wealth is moving forward"").

 eLDee has a masters degree in Architecture from the University of Lagos, Nigeria.

 He is an original member of the band Trybesmen, which was started in 1998 with rappers KB and Freestyle.

 He hails from Lagos Island in Lagos State of Nigeria.
</item 10>
=== END ITEM id=81edf131fcc7079e ===

=== ITEM id=11ce051f68441780 ===
Gold sentences and facts:
Sentence 0:  His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".
  Fact 0.0: The unnamed subject's American superior was General Dwight D. Eisenhower.
  Fact 0.1: Dwight D. Eisenhower wrote that the unnamed subject was 'blunt, at times to the point of rudeness'.
Sentence 1: Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United States from 1953 until 1961.
  Fact 1.0: Dwight David 'Ike' Eisenhower was born on October 14, 1890.
  Fact 1.1: Dwight David 'Ike' Eisenhower died on March 28, 1969.
  Fact 1.2: Dwight David 'Ike' Eisenhower was an American politician.
  Fact 1.3: Dwight David 'Ike' Eisenhower was an Army general.
  Fact 1.4: Dwight David 'Ike' Eisenhower served as the 34th President of the United States.
  Fact 1.5: Dwight David 'Ike' Eisenhower served as President from 1953 until 1961.

Retrieved items:
<item 1>
Joseph R. Anderson
Joseph Reid Anderson (February 16, 1813 – September 7, 1892) was an American civil engineer, industrialist, and soldier.

 During the American Civil War he served as a Confederate general, and his Tredegar Iron Company was a major source of munitions and ordnance for the
Confederate States Army.

 Starting with a small forge and rolling mill in the mid-1830s, It was a flourishing operation by 1843 when he leased it.

 He eventually bought the company outright in 1848 and forcefully and aggressively built Tredegar Iron Works into the South's largest and most significant iron
works.

 When the Civil War broke out he entered the Army as a Brigadier General in 1861.

 Shortly after he was wounded and then resigned from the Army returning to the iron works.

 It was the Confederacy's major (and for much of the war only) source of cannons and munitions, employing some 900 workers, most of whom slaves.

 His plant was confiscated by the United States In 1865, but returned to him in 1867 and he remained president until his death.

 Anderson was very active in local civic and political affairs.
</item 1>

<item 2>
Dwight D. Eisenhower
Dwight David "Ike" Eisenhower ( ; October 14, 1890 – March 28, 1969) was an American politician and Army general who served as the 34th President of the United
States from 1953 until 1961.

 During World War II, he was a five-star general in the United States Army and served as Supreme Commander of the Allied Expeditionary Forces in Europe.

 He was responsible for planning and supervising the invasion of North Africa in Operation Torch in 1942–43 and the successful invasion of France and Germany in
1944–45 from the Western Front.
</item 2>

<item 3>
John Anderson (Maine politician)
John Anderson (1792–1853) was a Maine politician.

 Anderson served as United States Representative from Maine from 1825-1833.
</item 3>

<item 4>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 4>

<item 5>
Benedict Arnold
Benedict Arnold (January 14, 1741 [O.S. January 3, 1740] June 14, 1801) was a general during the American Revolutionary War, who fought for the American
Continental Army, and later defected to the British Army.

 While a general on the American side, he obtained command of the fortifications at West Point, New York (which after 1802 would become the site of the U.S.
Military Academy), overlooking the cliffs at the Hudson River (upriver from British-occupied New York City), and planned to surrender them to British forces.

 This plan was exposed in September 1780.

 He was commissioned into the British Army as a brigadier general.
</item 5>

<item 6>
Wendell Anderson
Wendell Richard "Wendy" Anderson (February 1, 1933 – July 17, 2016) was an American politician and the 33rd governor of Minnesota, serving from January 4, 1971,
to December 29, 1976.

 In late 1976, he resigned as governor in order to be appointed to the U.S. Senate after Senator Walter Mondale was elected Vice President of the United States.

 Anderson served in the Senate from December 30, 1976, to December 29, 1978 (after losing the 1978 Senate election to Rudy Boschwitz, he resigned a few days
before the end of his term to give Boschwitz seniority).
</item 6>

<item 7>
The text profiles military leaders across the U.S. and Allied services. James Winnefeld was U.S. vice chairman of the Joint Chiefs (2011–15); Matthew Ridgway
led airborne forces in WWII; Arthur Tedder and naval officers Arthur Power and Algernon Willis helped plan or execute Allied Mediterranean campaigns. Hal Moore
and Myron Diduryk served at Ia Drang. George Finch built the Air National Guard; Mark Mitchell earned the Distinguished Service Cross in Afghanistan. Other
</item 7>

<item 8>
Andrew Davis Bruce
Lieutenant General Andrew Davis Bruce (September 14, 1894 – July 28, 1969) was an American academic and soldier who served as the third president of the
University of Houston.

 He retired from the United States Army in 1954 as a lieutenant general after seeing action in both World War I and World War II and founding Fort Hood, Texas.

 Three countries, France, the Philippines, and the United States, awarded him service medals, including the Distinguished Service Cross, the U.S. Army's second
highest military decoration.

 Bruce is interred in Arlington National Cemetery.
</item 8>

<item 9>
Kenneth Anderson (British Army officer)
General Sir Kenneth Arthur Noel Anderson, (25 December 1891 – 29 April 1959) was a senior British Army officer who saw service in both World wars.

 He is mainly remembered as the commander of the British First Army during Operation Torch, the Allied invasion of North Africa.

 He had an outwardly reserved character and did not court popularity either with his superiors or with the public.

 His American superior, General Dwight D. Eisenhower, wrote that he was "blunt, at times to the point of rudeness".

 In consequence he is less well known than many of his contemporaries.

 Richard Mead wrote that "He handled a difficult campaign more competently than his critics suggest, but competence without flair was not good enough for a top
commander in 1944".
</item 9>

<item 10>
John F. Anderson
John Francis Anderson was an officer in the Union Army during the American Civil War.

 When the war broke out, Anderson who was the son of Maine's former Governor Hugh J. Anderson was initially commissioned First Lieutenant in the militia; and on
September 2, 1861, Anderson became the adjutant of the 24th Regiment Massachusetts Volunteer Infantry.

 He was appointed to Maj. Gen. John G. Foster's staff as a major and an aide-de-camp on June 9, 1863 and served with that officer for most of the remainder of
the war, resigning on March 27, 1865.

 On December 11, 1866, President Andrew Johnson nominated Major Anderson to the honorary grade of brevet brigadier general, to rank from March 13, 1865 and the
United States Senate confirmed the award on February 6, 1867.
</item 10>
=== END ITEM id=11ce051f68441780 ===

=== ITEM id=3b642a32292ac164 ===
Gold sentences and facts:
Sentence 0: Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.
  Fact 0.0: Jack and Jill is an American magazine for children.
  Fact 0.1: Jack and Jill is published bimonthly.
  Fact 0.2: Jack and Jill is for children 7 to 12 years old.
  Fact 0.3: The title of Jack and Jill comes from the nursery rhyme of the same name.
Sentence 1: Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.
  Fact 1.0: Paper, stylized as PAPER, is a New York City-based magazine.
  Fact 1.1: Paper is an independent magazine.
  Fact 1.2: Paper focuses on fashion.
  Fact 1.3: Paper focuses on popular culture.
  Fact 1.4: Paper focuses on nightlife.
  Fact 1.5: Paper focuses on music.
  Fact 1.6: Paper focuses on art.
  Fact 1.7: Paper focuses on film.

Retrieved items:
<item 1>
Shape (magazine)
Shape is a monthly English language fitness magazine started by Weider Publications in 1981, founded by Christine MacIntyre (a pioneer in women's free weight
fitness) and became the number one women's fitness magazine.

 At that time, Weider Enterprises consisted primarily of the bodybuilding magazine "Muscle & Fitness".

 Joe Weider and Christine MacIntyre had differing views of how to present "Shape", Weider endorsing a less journalistic and more commercial approach to
articles, MacIntyre endorsing a more academic, doctor-based magazine.

 Weider also endorsed a sexier approach to editorial while MacIntyre endorsed a healthier look for women, eschewing sexiness in the models and the copy.

 MacIntyre largely won that battle, editing a magazine that required that every byline have an advanced medical degree, that cover models should look healthy
rather than sexy, and that sexist language be avoided.

 Christine MacIntyre was the editor-in-chief until her death in 1988.

 Tara Kraft is the current editor-in-chief.

 "Shape" found a readership based on that formula.
</item 1>

<item 2>
Paper (magazine)
Paper, stylized as PAPER, is a New York City-based independent magazine focusing on fashion, popular culture, nightlife, music, art and film.

 Past cover models include Kim Kardashian, Demi Lovato, Katy Perry, Miley Cyrus, Prince, CL, and Jennifer Lopez.
</item 2>

<item 3>
VIVmag
VIVmag is one of the earliest exclusively digital, interactive women's lifestyle magazines.

 Launched in 2006 and backed by Canadian industrialist David Harrison Gilmour (the founder of FIJI Water LLC), the bimonthly magazine is distributed by Zinio
Systems, Inc.

 The first issue appeared in January-February 2007.

 There is no print version.

 The paperless title can be viewed in most browsers or downloaded to your computer and read via the Zinio Reader.

 The magazine is based in Thousand Oaks, California.
</item 3>

<item 4>
Jack and Jill (magazine)
Jack and Jill is an American bimonthly magazine for children 7 to 12 years old that takes its title from the nursery rhyme of the same name.

 It features stories and educational activities.
</item 4>

<item 5>
Golf Digest
Golf Digest is a monthly golf magazine published by Condé Nast Publications in the United States.

 It is a generalist golf publication covering recreational golf and men's and women's competitive golf.

 Condé Nast Publications also publishes the more specialized "Golf for Women", "Golf World" and "Golf World Business".

 The magazine started in 1950, and was sold to The New York Times Company in 1969.

 The Times company sold their magazine division to Condé Nast in 2001.

 The headquarters of "Golf Digest" is in Des Moines, Iowa.
</item 5>

<item 6>
SexIs
SexIs Magazine is a quarterly print publication and daily webzine devoted to sex and sexual culture, founded in 2008 by Web Merchants, parent company of sex toy
e-tailer EdenFantasys.com.

 The first print issue debuted in November 2009, distributed nationwide as an insert in "BUST magazine".

 The website publishes articles, columns, video presentations and news items daily.
</item 6>

<item 7>
Essence (magazine)
Essence is a monthly magazine for African American women between the ages of 18 and 49.

 It is the only magazine that focuses on reaching an audience of black women, revolves around the black woman experience, and has remained for a long period of
time.

 The magazine covers fashion, lifestyle and beauty, with an intimate girlfriend-to-girlfriend tone, and its slogan "Fierce, Fun, and Fabulous" suggests the
magazine's goal of empowering African-American women.

 The topics the magazine discusses range from celebrities, to fashion, to point-of-view pieces addressing current issues in the African-American community.

 A number of its readers engage closely and personally with the publication, and it claims to be the magazine "for and about Black women".
</item 7>

<item 8>
Mighty Jill Off
Mighty Jill Off is a 2D platform independently developed freeware video game designed by Anna Anthropy, with art by James Harvey and music by Andrew Toups.

 It stars a submissive named Jill, who has a boot fetish and is forced to climb up a tower after her Queen kicks her down it as punishment.

 Jill does this by jumping and slowly descending over obstacles.

 Jill can be defeated in one hit by these obstacles, but will return to the last check point.

 The game serves as an homage to the 1986 arcade game "Mighty Bomb Jack".

 It had follow-ups, such as "Mighty Jill Off - Jill Off Harder Edition" and "Jill Off With One Hand".

 Jill made a cameo appearance in the 2010 video game "Super Meat Boy" as a playable character.
</item 8>

<item 9>
Jack &amp; Diane (film)
Jack & Diane is a 2012 American romantic horror film written and directed by Bradley Rust Gray, and starring Riley Keough and Juno Temple.

 Olivia Thirlby and Ellen Page were originally cast for the lead roles in 2008 before the project was postponed.
</item 9>

<item 10>
PC Magazine
PC Magazine (shortened as PCMag) is an American computer magazine published by Ziff Davis.

 A print edition was published from 1982 to January 2009.

 Publication of online editions started in late 1994 and continues to this day.
</item 10>
=== END ITEM id=3b642a32292ac164 ===

=== ITEM id=06eb3c2dd6a5a0d3 ===
Gold sentences and facts:
Sentence 0:  It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
  Fact 0.0: This song is most known through the 1989 single by Kathy Mattea.
  Fact 0.1: The 1989 single by Kathy Mattea was released in conjunction with her album Willow in the Wind.
  Fact 0.2: This song was first recorded and released on the 1987 Don Williams album Traces.
  Fact 0.3: This song was also released in 1988 by Clark's husband on his album Old Friends.
Sentence 1: This is a detailed discography for American country music artist Don Williams.
  Fact 1.0: The discography is detailed.
  Fact 1.1: The discography is for American country music artist Don Williams.

Retrieved items:
<item 1>
The entries cover songs and recordings spanning R&B, rock, pop, and folk. Highlights include Johnny Otis’s 1958 “Willie and the Hand Jive” (US #9), later
covered by Eric Clapton (#26) and George Thorogood; Van Halen’s “Finish What Ya Started,” co-written overnight by Eddie Van Halen and Sammy Hagar; and Johnny
Otis’s “Every Beat of My Heart,” later a major hit for the Pips.
</item 1>

<item 2>
Night Is Fallin' in My Heart
"Night Is Fallin' in My Heart" is a song written by Dennis Linde.

 The song was first recorded by Country music artist J.P. Pennington for his 1991 album, "Whatever It Takes".

 It was later recorded and released as a single in October 1994 by American country music group Diamond Rio.

 It was the second single released from their third album, "Love a Little Stronger".

 It peaked at number 9 in the United States, and number 6 in Canada.
</item 2>

<item 3>
Hjärtats trakt
Hjärtats trakt was released on 20 April 1993 and is a compilation album from Swedish pop artist Per Gessle.

 The album includes songs taken from Gessle's solo albums "Per Gessle" and "Scener".

 It was also released under license by the British record company Pickwick and it is the only Gessle album released by another label than EMI.

 In 1998 the album was re-released by the Dutch record company Disky Communications B.V. which is a subsidiary of EMI.
</item 3>

<item 4>
The entries cover diverse music milestones: Marc Ribot’s 2005 Albert Ayler tribute, *Spiritual Unity*; comic duo Collins & Harlan, whose 1916 recording helped
introduce “jas,” precursor to “jazz”; and reggae singer Cedric Myton, a Royal Rasses and Congos member whose voice features on *Humanity* and *Heart of the
Congos*. They also describe early Dean Martin, Chico Hamilton, Shelly Manne, and Maynard
</item 4>

<item 5>
Come On (Earl King song)
"Come On" (often called "Let the Good Times Roll") is a song written by New Orleans rhythm and blues artist Earl King.

 He first recorded the song as "Darling Honey Angel Child" in 1960 for the Ace Records subsidiary Rex.

 Later that year, he recorded it as a two-part song for Imperial Records using some new lyrics.

 Retitled "Come On", it was released in 1960 with "Come On - Part I” as the A-side backed with “Come On - Part II” (Imperial 5713).
</item 5>

<item 6>
Come from the Heart
"Come from the Heart" is a country music song written by Richard Leigh and Susanna Clark and published in 1987.

 It is most known through the 1989 single by Kathy Mattea, released in conjunction with her album "Willow in the Wind", though the song was first recorded and
released on the 1987 Don Williams album "Traces" and also released in 1988 by Clark's husband on his album "Old Friends".
</item 6>

<item 7>
Hate to See Your Heart Break
"Hate to See Your Heart Break" is a song by American rock band Paramore, recorded for their 2013 self-titled fourth album "Paramore".

 It was re-recorded to feature vocals by Joy Williams (formerly of The Civil Wars) for the 2014 deluxe edition of the album, the first time Paramore has
collaborated with another artist or group on a studio recording.
</item 7>

<item 8>
Every Beat of My Heart (song)
"Every Beat of My Heart" is a rhythm and blues song by Johnny Otis.

 It was first recorded in 1954 by his group, The Royals (later to be known as The Midnighters).

 In 1961 Gladys Knight & the Pips recorded the song for their debut single on the Vee-Jay label.

 Credited to The Pips, it was the first of eleven releases by the group to make it to number one on the R&B/soul chart.

 It was also the group's first top ten on the "Billboard" Hot 100.

 "Every Beat of My Heart" was first recorded for the Huntom label, who later sold the master to Vee-Jay.

 At the time of the song's release, The Pips were on the Fury label where they re-recorded the song without piano.

 In an unusual occurrence, the Fury recording of the song also made the top twenty on the R&B Sides chart and also made the Hot 100.
</item 8>

<item 9>
Streets of Heaven (song)
"Streets of Heaven" is a song co-written and recorded by Australian country music artist Sherrié Austin.

 It was released in June 2003 as the first single and title track from her album "Streets of Heaven".

 Austin wrote the song with Paul Duncan and Al Kasha.
</item 9>

<item 10>
Profiles cover pop, R&B, country, and rock releases: Tevin Campbell’s Grammy-nominated “Can We Talk” topped the US R&B chart; Steve Winwood’s “Back in the High
Life Again” reached No. 1 Adult Contemporary, while *Refugees of the Heart* featured a chart-topping rock single. Other highlights include LeAnn Rimes’s
“Looking Through Your Eyes,” Westlife’s UK No. 1 debut “Swear It Again,” and Mar
</item 10>
=== END ITEM id=06eb3c2dd6a5a0d3 ===
