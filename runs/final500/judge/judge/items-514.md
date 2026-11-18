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

=== ITEM id=733fd7ffa3e092b8 ===
Gold sentences and facts:
Sentence 0:  It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
  Fact 0.0: The film featured Reese Witherspoon and Paul Rudd.
  Fact 0.1: Reese Witherspoon and Paul Rudd were in the film before both became considerably bigger film stars.
Sentence 1:  Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for which she was nominated for a Young Artist Award.
  Fact 1.0: The actress was born in New Orleans.
  Fact 1.1: The actress was raised in Tennessee.
  Fact 1.2: The actress began her career as a child actress.
  Fact 1.3: The actress made her professional screen debut in 'The Man in the Moon' (1991).
  Fact 1.4: The actress was nominated for a Young Artist Award for her debut in 'The Man in the Moon'.

Retrieved items:
<item 1>
Rachelle Lefevre
Rachelle Lefevre ( ; born February 1, 1979) is a Canadian actress.

 She has starred in the television series "Big Wolf on Campus" and had recurring roles in "What About Brian", "Boston Legal", and "Swingtown".

 She played the vampire Victoria Sutherland in the first two films of the "Twilight" saga, before being replaced by Bryce Dallas Howard in the third
installment, "", due to scheduling conflicts with her role in the film "Barney's Version".

 In 2011, she starred in the ABC medical drama "Off the Map", followed by the CBS series "A Gifted Man" (2011–2012) and "Under the Dome" (2013–2015).
</item 1>

<item 2>
Overnight Delivery
Overnight Delivery is a 1998 romantic comedy film directed by Jason Bloom.

 It was rated PG-13 by the MPAA and released direct-to-video.

 It featured Reese Witherspoon and Paul Rudd, prior to both becoming considerably bigger film stars.
</item 2>

<item 3>
The Night Before (1988 film)
The Night Before is a 1988 film starring Keanu Reeves and Lori Loughlin.

 Reeves plays Winston Connelly, the so-called high school nerd and vice president of the astronomy club.

 Loughlin plays Tara Mitchell, the pretty and popular head cheerleader who also happens to be the local police chief's daughter.

 The Tagline was: "You lost your father's car, sold your prom date and a guy called 'Tito' wants you dead.

 It's a date that's the time of your life."

 It was filmed and set entirely in Los Angeles, California.
</item 3>

<item 4>
Middle of Nowhere (2012 film)
Middle of Nowhere is a 2012 independent feature film written and directed by Ava DuVernay and starring Emayatzy Corinealdi, David Oyelowo, Omari Hardwick and
Lorraine Toussaint.

 The film was the winner of the Directing Award for U.S. Dramatic Film at the 2012 Sundance Film Festival.
</item 4>

<item 5>
Reese Witherspoon
Laura Jeanne Reese Witherspoon ( ; born March 22, 1976) is an American actress, producer, and entrepreneur.

 Born in New Orleans and raised in Tennessee, she began her career as a child actress, making her professional screen debut in "The Man in the Moon" (1991), for
which she was nominated for a Young Artist Award.

 Following breakout roles in "" (1992) and "Jack the Bear" (1993), she starred in the comedy-drama "Pleasantville" (1998), for which she won the Young Hollywood
Award for Breakthrough Performance.

 Her leading role of Tracy Flick in "Election" (1999) was nominated for a Golden Globe Award.
</item 5>

<item 6>
The Notebook
The Notebook is a 2004 American romantic drama film directed by Nick Cassavetes and based on the 1996 novel of the same name by Nicholas Sparks.

 The film stars Ryan Gosling and Rachel McAdams as a young couple who fall in love in the 1940s.

 Their story is narrated from the present day by an elderly man (portrayed by James Garner) telling the tale to a fellow nursing home resident (played by Gena
Rowlands, who is Cassavetes's mother).
</item 6>

<item 7>
Zooey Deschanel filmography
American actress Zooey Deschanel made her film debut in the 1999 comedy feature "Mumford".

 She went on to gained public attention by co-starred in the comedy-drama "Almost Famous" (2000), the independent drama "Manic" (2001), opposite Joseph
Gordon-Levitt, and the comedy-drama "The Good Girl" (2002).

 She landed her first major role as a 18-year-old virgin in the romantic drama "All the Real Girls" (2003), for which she was nominated for an Independent
Spirit Award for Best Female Lead.

 Deschanel's biggest commercial success, as of 2017, came with the Christmas-fantasy film "Elf" (also in 2003), which grossed over $220 million worldwide.
</item 7>

<item 8>
Chloe (film)
Chloe is a 2009 erotic thriller film directed by Atom Egoyan, a remake of the 2003 French film "Nathalie...".

 It stars Julianne Moore, Liam Neeson, and Amanda Seyfried in the title role.

 Its screenplay was written by Erin Cressida Wilson, based on the earlier French film, written by Anne Fontaine.
</item 8>

<item 9>
Annie Belle
Annie Brilland (born 10 December 1956) is a French actress and social worker.

 Her acting career began in 1974 and throughout the seventies, has had a series of varied roles in both French and Italian cinema, working with such directors
as Jean Rollin, Ruggero Deodato and Joe D'Amato.
</item 9>

<item 10>
City of New Orleans (train)
The City of New Orleans is an Amtrak passenger train which operates on an overnight schedule between Chicago and New Orleans.

 The train is a successor to the Illinois Central Railroad's "Panama Limited".

 The present name was revived in 1981.
</item 10>
=== END ITEM id=733fd7ffa3e092b8 ===

=== ITEM id=cd0f9879c3f19f6a ===
Gold sentences and facts:
Sentence 0: The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.
  Fact 0.0: The Livesey Hall War Memorial commemorates the fallen of World War I.
  Fact 0.1: The Livesey Hall War Memorial commemorates the fallen of World War II.
  Fact 0.2: The commemorated fallen had been employed by the South Suburban Gas Company of London.
Sentence 1:  Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).
  Fact 1.0: Over 60 million people were killed.
  Fact 1.1: The deaths were about 3% of the 1940 world population.
  Fact 1.2: The estimated 1940 world population was 2.3 billion.

Retrieved items:
<item 1>
Freetown Memorial
The Freetown Memorial is a Commonwealth War Graves Commission war memorial located outside the Secretariate Building in Freetown, Sierra Leone.

 The memorial generally commemorates Soldiers from Sierra Leone who died whilst serving with the Royal West African Frontier Force in West Africa and whose
graves are not known.

 The Freetown Memorial commemorates 1,109 First World War casualties and 247 from the Second World War.
</item 1>

<item 2>
Rainham War Memorial
The Rainham War Memorial commemorates soldiers killed in both World Wars as well as civilian casualties of World War II.
</item 2>

<item 3>
Vietnam War Memorial, Hanoi
The War Memorial in Hanoi is located across the Ba Dinh Square, across the Ho Chi Minh Mausoleum and close to Hanoi Citadel.

 Constructed in 1993 in a fusion of traditional Vietnamese and modernist architecture, the memorial commemorates men and women who sacrificed themselves during
the Second Indochina War.

 The war is known by many names, e.g. as the American War in Vietnam.

 The memorial is a focal point for state functions, commemorating the war dead.
</item 3>

<item 4>
Tower Hill Memorial
The Tower Hill Memorial is a Commonwealth War Graves Commission war memorial on the south side of Trinity Square Gardens, in London, England.

 The memorial commemorates those from the Merchant Navy and fishing fleets who died during both world wars and have "no grave but the sea".

 The memorial was designed by Edwin Lutyens with sculpture work by William Reid Dick, the Second World War extension was designed by Edward Maufe with sculpture
work by Charles Wheeler.
</item 4>

<item 5>
Livesey Hall War Memorial
The Livesey Hall War Memorial commemorates the fallen of World War I and World War II who had been employed by the South Suburban Gas Company of London.

 It is also a tribute to those employees who served in the wars.

 The monument was designed and executed by British sculptor Sydney March, of the March family of artists.
</item 5>

<item 6>
World War II casualties
World War II was the deadliest military conflict in history in absolute terms of total casualties.

 Over 60 million people were killed, which was about 3% of the 1940 world population (est. 2.3 billion).

 The tables below give a detailed country-by-country count of human losses.

 World War II fatality statistics vary, with estimates of total deaths ranging from 50 million to more than 80 million.

 The higher figure of over 80 million includes deaths from war-related disease and famine.

 Civilians killed totalled 50 to 55 million, including 19 to 28 million from war-related disease and famine.

 Military deaths from all causes totalled 21 to 25 million, including deaths in captivity of about 5 million prisoners of war.
</item 6>

<item 7>
Portsmouth Naval Memorial
The Portsmouth Naval Memorial, sometimes known as Southsea Naval Memorial, is a war memorial in Portsmouth, Hampshire, England, on Southsea Common beside
Clarence Esplanade, between Clarence Pier and Southsea Castle.

 The memorial commemorates approximately 25,000 British and Commonwealth sailors who were lost in the World Wars, around 10,000 sailors in the First World War
and 15,000 in the Second World War.

 The memorial features a central obelisk, with names of the dead on bronze plaques arranged around the memorial according to the year of death.
</item 7>

<item 8>
Arras Flying Services Memorial
The Arras Flying Services Memorial Commonwealth War Graves Commission war memorial in the Faubourg d'Amiens Cemetery, Arras, France.

 The memorial commemorates nearly 1,000 airmen from forces of the Commonwealth who were killed on the Western Front during World War I and who have no known
grave.

 The memorial was designed by Edwin Lutyens, sculpted by William Reid Dick and unveiled by Hugh Trenchard, 1st Viscount Trenchard, Marshal of the Royal Air
Force on 31 July 1932.
</item 8>

<item 9>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 9>

<item 10>
Alamein Memorial
The Alamein Memorial is a Commonwealth War Graves Commission war memorial in the El Alamein War Cemetery, El Alamein, Egypt.

 The memorial commemorates 11,866 Commonwealth forces members who died during World War II.

 The memorial was designed by Hubert Worthington and unveiled by Viscount Montgomery of Alamein on 24 October 1954.
</item 10>
=== END ITEM id=cd0f9879c3f19f6a ===

=== ITEM id=8bc736dd659c689d ===
Gold sentences and facts:
Sentence 0: The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.
  Fact 0.0: The Night of the Arts is an annual event.
  Fact 0.1: The Night of the Arts is known in Finnish as 'Taiteiden yö'.
  Fact 0.2: The Night of the Arts is known in Swedish as 'Konstens natt'.
  Fact 0.3: The Night of the Arts is held in several major cities in Finland.
  Fact 0.4: The Night of the Arts is held in late August.
Sentence 1:  It is one of many White Night festivals held worldwide.
  Fact 1.0: The Night of the Arts is one of many White Night festivals.
  Fact 1.1: White Night festivals are held worldwide.
Sentence 2: The White Nights are a kind of all-night arts festival held in many cities in the summer.
  Fact 2.0: The White Nights are a kind of all-night arts festival.
  Fact 2.1: The White Nights festival is held in many cities.
  Fact 2.2: The White Nights festival is held in the summer.
Sentence 3:  The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are early and darkness is never complete.
  Fact 3.0: 'White nights' is the name given in areas of high latitude to the weeks around the summer solstice in June.
  Fact 3.1: During white nights, sunsets are late.
  Fact 3.2: During white nights, sunrises are early.
  Fact 3.3: During white nights, darkness is never complete.

Retrieved items:
<item 1>
Joulupöytä
Joulupöytä (translated "Yule table") is the traditional assortment of foods served at Christmas in Finland, similar to the Swedish julbord.

 It contains many different dishes, most of them typical for the season.

 The main dish is usually a large Christmas ham, which is eaten with mustard or bread along with the other dishes.

 Fish is also served (often lutefisk and gravlax), and the ham is served with "laatikko"s, casseroles made with swede, potato and carrot, occasionally liver.

 The traditional Christmas beverage is either alcoholic or non-alcoholic mulled wine ("glögi" in Finnish).
</item 1>

<item 2>
Kerala School Kalolsavam
School Arts Festival of Kerala (Malayalam: കേരള സ്കൂൾ കലോത്സവം) is an annual event conducted by the state government of Kerala, featuring several art
competitions for high school and higher secondary school students of Kerala.

 The festival was started in 1956, and till 2008, it was called as "Kerala State School Youth Festival".

 The participants are students from classes 8th to 12th.

 Winners from different revenue districts for a particular event will be competing in state level competition.

 The event is usually conducted in December–January months of a year and is considered to be the biggest cultural event of Asia.

 57th edition of Kerala School Kalolsavam 2017 is scheduled for January 16 to 22 at Kannur.
</item 2>

<item 3>
White Night festivals
The White Nights are a kind of all-night arts festival held in many cities in the summer.

 The original festival is the White Nights Festival held in Saint Petersburg, Russia.

 The "white nights" is the name given in areas of high latitude to the weeks around the summer solstice in June during which sunsets are late, sunrises are
early and darkness is never complete.

 In Saint Petersburg, the Sun does not set until after 10 p.m., and the twilight lasts almost all night.
</item 3>

<item 4>
Artscape (festival)
Artscape is an annual art festival held in the Mount Royal neighborhood of Baltimore, Maryland in July.

 Since its first annual event in 1982, it has become the largest free arts festival in America.

 It has boasted acts such as Ray Charles, Aretha Franklin, and Matisyahu in the past, attracting over 350,000 people from the city, and surrounding areas.

 Film programming during Artscape is provided by Maryland Film Festival.

 There are artists in a variety of visual and performing media.

 Events are free and open to the public.
</item 4>

<item 5>
Green Man Festival
Green Man is an independent music and arts festival held annually in mid-August in the Brecon Beacons, Wales since 2003.

 It has evolved into a 20,000 capacity 4-day event, showcasing predominantly live music (in particular alternative, indie, rock, folk, dance and americana),
with additional events showcasing literature, film, comedy, theatre and poetry.

 2015 saw 1,500 multi-arts acts perform across 17 stages.

 The festival site is divided into 10 areas, each offering a unique festival experience.

 Ceilidhs, all-night bonfires and secret gigs all add to the festival's unique identity.
</item 5>

<item 6>
Flip Animation Festival
FLIP is an animation festival primarily hosted by the Light House Media Centre in Wolverhampton, UK.

 It is one of two festivals hosted by Light House, the other of which is Deaffest.

 Official literature for the festival says that FLIP occurs annually at the beginning of November and attracts submissions from more than 30 countries world
wide.

 As well as screening the selected open submissions, FLIP also consists of special screenings, talks from professionals within the animation world, workshops,
industry panels, portfolio reviews and competitions.

 The festival was set up, managed and programmed by Peter McLuskie between 2004 and 2011.

 It grew out of the 'Animation Forum', also based at Light House and which was later rebranded as Animation Forum West Midlands and found a home at Birmingham
City University.

 In 2009 the festival was awarded a Black Country Tourism Award for Event of the Year.
</item 6>

<item 7>
Great Northern Arts Festival
The Great Northern Arts Festival is held each year in Inuvik, Northwest Territories.

 Lasting approximately ten days, it is an annual event held each summer.

 It was established in 1989 by Sharlene Alexander and Sue Rose.

 While the festival's mandate is to serve Northern artists and craftspeople, artists from around the world also attend.

 The festival is designed as a venue for artists to see each other's work and to experiment with new applications.

 It includes arts of all types, such as fiber art and Music of the Northwest Territories.

 Artists partake in demonstrations, masterclasses, seminars, and workshops.

 The festival's gallery offers pieces for sale.

 There are cultural presentations each evening which include concerts, dance, fashion, music, and storytelling.

 Media coverage brings festival highlights to viewing audiences.
</item 7>

<item 8>
Tanglewood Music and Arts Festival
Tanglewood Music and Arts Festival is an annual two-day music, comedy and arts event held within the grounds of Narrow Water Castle, County Down, Northern
Ireland.

 The festival, founded in 2011, has attracted significant attention both within the local area and around Northern Ireland with buses coming from as far as
Belfast to the event.

 For the 2011 and 2012 festivals Tanglewood was held on one day; however for 2013 it has expanded into a two-day camping festival.
</item 8>

<item 9>
Scandinavian folklore
Scandinavian folklore or Nordic folklore is the folklore of Norway, Sweden, Finland, Denmark, Iceland and the Faroe Islands.
</item 9>

<item 10>
Night of the Arts
The Night of the Arts (Finnish: "Taiteiden yö" , Swedish: "Konstens natt" ) is an annual event held in several major cities in Finland, in late August.

 It is one of many White Night festivals held worldwide.
</item 10>
=== END ITEM id=8bc736dd659c689d ===

=== ITEM id=b87344e205f3f054 ===
Gold sentences and facts:
Sentence 0:  Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
  Fact 0.0: Cacareco's candidacy inspired the Rhinoceros Party of Canada.
  Fact 0.1: The Rhinoceros Party of Canada was nominally led by the rhinoceros Cornelius the First.
Sentence 1: The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.
  Fact 1.0: The Rhinoceros Party's French name is 'Parti Rhinoceros'.
  Fact 1.1: The Rhinoceros Party was a registered political party in Canada.
  Fact 1.2: The Rhinoceros Party was a registered political party in Canada from the 1960s to the 1990s.

Retrieved items:
<item 1>
National Advancement Party of Canada
The National Advancement Party of Canada (French: "Parti de l'Essor National du Canada" ) is a minor federal political party in Canada.

 The party describes itself as for "advancement of the Canadian people and Canada".

 The party was formed in 2014 and registered with Elections Canada in 2015, initially under the name Democratic Advancement Party of Canada.

 The party is headquartered in Calgary, Alberta and led by writer Stephen Garvey.
</item 1>

<item 2>
Canada grew from British North American colonies into a federation in 1867; Newfoundland and Labrador joined in 1949. The Statute of Westminster (1931) granted
legislative sovereignty, completed by the Canada Act (1982). The Stamp Act taxed colonial printed materials, prompting the 1765 Stamp Act Congress. The West
Indies Federation (1958–62) collapsed amid political disputes. Other topics include the Royal Mint, archaeological hoards, Canadian file sharing, Wood Green’s
war memorial, Québec
</item 2>

<item 3>
Rhinoceros Party of Canada (1963–93)
The Rhinoceros Party (French: "Parti Rhinocéros") was a registered political party in Canada from the 1960s to the 1990s.

 Operating within the tradition of political satire, the Rhinoceros Party's basic credo, their so-called primal promise, was "a promise to keep none of our
promises".

 They then promised outlandishly impossible schemes designed to amuse and entertain the voting public.
</item 3>

<item 4>
Aboriginal Peoples Party of Canada
The Aboriginal Peoples Party of Canada (APP) is a Canadian political party that was founded in 2005.

 The party was conceived by University of Lethbridge student Myron Wolf Child.

 It held its founding meeting on August 21, 2005, in St. Albert, Alberta.

 The APP was headed by interim leader Bill Montour, a former Chief of the Six Nations Reserve in Ontario.

 In October 2005, the party had 122 confirmed members, just less than half the number needed to register as an official party in Canada.

 In late October 2005, the APP sought to unite with the First Peoples National Party of Canada which also had fewer than the number of confirmed members needed
to become a registered political party.

 The First Peoples National Party of Canada became an eligible political party on December 6, 2005.

 Whether or not this was accomplished through a merger with the APP is unclear.

 If this is the case, the APP no longer exists, but, as one of the stipulations of a possible merger was that a name for the new party would be determined at
the party's first convention, it is possible that the name Aboriginal Peoples Party of Canada will come into usage again.
</item 4>

<item 5>
Revolutionary Communist Party of Canada
The Revolutionary Communist Party of Canada, or RCP Canada, is a revolutionary communist party oriented around Marxism-Leninism-Maoism.

 The creation of the organizational stage of the Party was adopted at what was called the "Revolutionary Communist Conference", which was held in Montreal,
Quebec in November 2000 by activists and former members of the labour union movements and youth organizations of the 1960s, 1970s and 1980s, who felt that the
revolutionary situation in Canada warranted the creation of a Party dedicated to a communist revolution.

 The party was initially called the Revolutionary Communist Party (Organizing Committees) (RCP(OC)).

 At this conference, participants adopted the Party's first Draft Programme.

 Because of the location of the conference, the majority of the founding members were French-speaking Quebecers, and the Party began an extensive effort to
reach out to the rest of Canada, starting with the Canadian Revolutionary Congress held in November 2006 in Toronto.

 It is not recognized as a political party by Elections Canada, because the party rejects what it calls the "bourgeois electoral system" and doesn't seek
recognition by the State.
</item 5>

<item 6>
Cacareco
Cacareco (1954-1962) was a female black rhinoceros exhibited in Brazilian zoos.

 She became famous as a candidate for the 1958 São Paulo city council elections with the intention of protesting against political corruption.

 Electoral officials did not accept Cacareco's candidacy, but she eventually won 100,000 votes, more than any other party in that same election (which was also
marked by rampant absenteeism).

 Today, the term "voto Cacareco" (Cacareco vote) is commonly used to describe protest votes in Brazil.

 Cacareco's candidacy inspired the Rhinoceros Party of Canada, nominally led by the rhinoceros Cornelius the First.
</item 6>

<item 7>
Politics of Canada
The politics of Canada function within a framework of parliamentary democracy and a federal system of parliamentary government with strong democratic
traditions.

 Canada is a constitutional monarchy, in which the Monarch is head of state.

 The country has a multi-party system in which many of its legislative practices derive from the unwritten conventions of and precedents set by the United
Kingdom's Westminster Parliament.

 However, Canada has evolved variations: party discipline in Canada is stronger than in the United Kingdom and more parliamentary votes are considered motions
of confidence, which tends to diminish the role of non-Cabinet Members of Parliament, (MPs).

 Such members, in the government caucus, and junior or lower-profile members of opposition caucuses, are known as backbenchers. Backbenchers can, however, exert
their influence by sitting in parliamentary committees, like the Public Accounts Committee or the National-Defence Committee.
</item 7>

<item 8>
The New Canada
The New Canada is a Canadian political literature book written by Reform Party of Canada founder and leader Preston Manning and published by Macmillan Canada.

 The book explains the personal, religious, and political life of Preston Manning and explains the roots and beliefs of the Reform Party.

 At the time of its publishing in 1992, Reform had become a popular populist conservative party in Western Canada after the mainstream Progressive Conservative
Party of Canada was collapsing in support and in 1991 decided to expand eastward into Ontario and the Maritimes.

 One year later the PC party collapsed in the 1993 federal election, allowing the Reform Party to make political history in Canada, deplacing the PCs as the
dominant conservative party in Canada.

 Reform, later renamed the Canadian Alliance, merged with the PC Party in 2003, to form a united right-wing alternative to the governing Liberal Party of
Canada, named the Conservative Party of Canada which has dropped many of the populist themes that the Reform Party had.
</item 8>

<item 9>
LPRadicals, founded in 2006, opposed Libertarian platform deletions and promotes four strategic principles; active at conventions through 2016. Hong Kong’s CHRF
unites pan-democratic groups and organizes July 1 marches. The U.S. Republican Party arose in 1854 opposing the Kansas–Nebraska Act. Preston Manning’s 1992 book
traces his life and Reform Party’s rise and later merger into today’s Conservatives. Canada’s Aboriginal Peoples Party (2005)
</item 9>

<item 10>
Cabécar language
The Cabécar language is an indigenous American language of the Chibchan language family which is spoken by the Cabécar people in Costa Rica.

 Specifically, it is spoken in the inland Turrialba Region of the Cartago Province.

 80% of speakers are monolingual; as of 2007, it is the only indigenous language in Costa Rica with monolingual adults.

 The language is also known by its dialect names Chirripó, Estrella, Telire, and Ujarrás.
</item 10>
=== END ITEM id=b87344e205f3f054 ===

=== ITEM id=9f9db680fe19fcae ===
Gold sentences and facts:
Sentence 0: Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.
  Fact 0.0: Scott Richard Weiland was born Scott Richard Kline.
  Fact 0.1: Scott Richard Weiland was born on October 27, 1967.
  Fact 0.2: Scott Richard Weiland's date range ends on December 3, 2015.
  Fact 0.3: Scott Richard Weiland was an American musician.
  Fact 0.4: Scott Richard Weiland was an American singer.
  Fact 0.5: Scott Richard Weiland was an American songwriter.
Sentence 1: Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.
  Fact 1.0: Richard Willis Hawley was born on 17 January 1967.
  Fact 1.1: Richard Willis Hawley is English.
  Fact 1.2: Richard Willis Hawley is a guitarist.
  Fact 1.3: Richard Willis Hawley is a singer-songwriter.
  Fact 1.4: Richard Willis Hawley is a producer.

Retrieved items:
<item 1>
Scott Weiland and the Wildabouts
Scott Weiland and the Wildabouts (sometimes billed simply as The Wildabouts) were a rock and roll band formed sometime around 2012 by former Stone Temple Pilots
and Velvet Revolver lead singer Scott Weiland, along with bassist Tommy Baker, guitarist Jeremy Brown, and multi-instrumentalist Doug Grean.

 The Wildabouts were marred with several roster changes (including the death of guitarist Jeremy Brown) and ultimately disbanded after Weiland's sudden death on
a tour bus in late 2015.
</item 1>

<item 2>
Doug Grean
Doug Grean is an American record producer, audio engineer, guitarist, based in Los Angeles.

 He has worked with artists Sheryl Crow, Ricki Lee Jones, Glen Campbell, Scott Weiland, Slash, Velvet Revolver, Stone Temple Pilots, The Crystal Method, Tim
McGraw, Cyndi Lauper, Annabella Lwin (Bow Wow Wow), Corey Glover (Living Color), Camp Freddy, DJ Hurricane and John Taylor (Duran Duran).

 Grean was formerly the lead guitarist and touring musical director for Scott Weiland and the Wildabouts.

 Some of Grean's television appearances with Weiland include: VH1 Legends, Last Call with Carson Daly, The Tonight Show with Jay Leno, and the Grammy awards
with Velvet Revolver as keyboardist, performing "All Across the Universe" with Stevie Wonder, Norah Jones, Bono, Steven Tyler, Brian Wilson, and Alison Krauss.
</item 2>

<item 3>
Warrel Dane
Warrel Dane (born March 7, 1969) is the lead singer of the American power metal band Sanctuary.

 He is also known as the former lead singer of currently inactive progressive metal band Nevermore.

 He is a natural baritone, though he was known for his high-pitched vocals with Serpent's Knight and on the first two Sanctuary albums.

 Later in his career, Dane became more notable for his distinctively deep, dramatic voice.
</item 3>

<item 4>
Standing at the Sky's Edge
Standing at the Sky's Edge is the seventh studio album from English musician Richard Hawley, released in the UK on 7 May 2012 and in the US on 28 August 2012.

 The album is markedly different from Hawley's previous efforts, often relinquishing softer instrumentation in favour of squalling guitars.

 As with all of Hawley's previous albums, the title obliquely refers to a location in his native city of Sheffield, in this case Skye Edge, a hillside area with
views over the city centre and formerly known for its crime-ridden estates but largely redeveloped in the mid-2000s.
</item 4>

<item 5>
&quot;Happy&quot; in Galoshes
"Happy" in Galoshes is the second solo album by American rock singer Scott Weiland.

 Weiland, known for his roles as the lead singer in Stone Temple Pilots and Velvet Revolver, released his first album, "12 Bar Blues" in 1998.

 Ten years later, ""Happy" in Galoshes" served as the official follow-up.

 Two versions have been released, a single disc and double disc deluxe version—the deluxe edition offers a second disc of 10 extra tracks.

 The album was released November 25, 2008 on Weiland's own Softdrive Records.

 Produced by Doug Grean and Scott Weiland, with select tracks recorded by Steve Albini, the album features guest appearances by Paul Oakenfold and No Doubt
members Adrian Young, Tony Kanal, and Tom Dumont.

 Oakenfold appears on a cover of David Bowie's "Fame."

 Weiland has cited Bowie as one of his main influences.
</item 5>

<item 6>
Scott Weiland
Scott Richard Weiland ( ; born Scott Richard Kline; October 27, 1967 – December 3, 2015) was an American musician, singer and songwriter.

 During a career spanning three decades, Weiland was best known as the lead singer of the band Stone Temple Pilots from 1989 to 2002 and 2008 to 2013.

 He was also a member of supergroup Velvet Revolver from 2003 to 2008 and recorded one album with another supergroup, Art of Anarchy.

 He established himself as a solo artist as well, releasing three studio albums, two cover albums, and collaborations with several other musicians throughout
his career.
</item 6>

<item 7>
A Compilation of Scott Weiland Cover Songs
A Compilation of Scott Weiland Cover Songs is a compilation album and third solo studio album by Stone Temple Pilots frontman Scott Weiland, released on August
30, 2011.

 The album consists entirely of cover songs, featuring an array of covers of artists that inspired Weiland, such as David Bowie, The Beatles, The Rolling
Stones, and The Smiths.

 The album was originally set to be released together with Weiland's memoir "Not Dead & Not for Sale" but Weiland decided later to release the album by itself
exclusively in digital format.
</item 7>

<item 8>
Steve Winwood
Stephen Lawrence "Steve" Winwood (born 12 May 1948) is an English musician whose genres include rock, blue-eyed soul, rhythm and blues, blues rock, pop rock,
and jazz.

 Though primarily a vocalist and keyboardist, Winwood also plays bass guitar, drums, acoustic and electric guitar, mandolin, violin, and other strings.
</item 8>

<item 9>
Richard Hawley
Richard Willis Hawley (born 17 January 1967) is an English guitarist, singer-songwriter and producer.

 After his first band Treebound Story (formed while he was still at school) broke up, Hawley found success as a member of Britpop band Longpigs in the 1990s.

 After that group broke up in 2000, he later joined the band Pulp, led by his friend Jarvis Cocker, for a short time.

 As a solo musician, Hawley has released seven studio albums.

 He has been nominated for a Mercury prize twice and once for a Brit Award.

 He has collaborated with Lisa Marie Presley, Arctic Monkeys, Manic Street Preachers and Paul Weller.
</item 9>

<item 10>
Hugh Whitaker
Hugh Whitaker (born 18 May 1961) is an English musician and the former drummer for the British indie rock band The Housemartins.

 He replaced original drummer Chris Lang and drummed for the band's first album, "London 0 Hull 4", and its attendant single releases.

 He left the band before the recording of their second album, "The People Who Grinned Themselves to Death".

 Whitaker left the band on amicable terms and even participated in the promotional video for the band's first single without him, "Five Get Over Excited",
wherein he was kidnapped by his replacement, Dave Hemingway, and locked in a hessian sack.
</item 10>
=== END ITEM id=9f9db680fe19fcae ===

=== ITEM id=9ce0ca89d16b8230 ===
Gold sentences and facts:
Sentence 0: Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.
  Fact 0.0: Jürgen Melzer was born on 22 May 1981.
  Fact 0.1: Jürgen Melzer was born in Vienna.
  Fact 0.2: Jürgen Melzer is an Austrian tennis player.
Sentence 1: Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
  Fact 1.0: Tom Gullikson was born on September 8, 1951.
  Fact 1.1: Tom Gullikson is a tennis coach.
  Fact 1.2: Tom Gullikson is a former professional tennis player.
  Fact 1.3: Tom Gullikson is from the United States.

Retrieved items:
<item 1>
Julian Knowle
Julian Knowle (born 29 April 1974) is an Austrian male professional tennis player.

 Being a born left-hander, Knowle is now one of the few on the ATP Tour who plays his forehand, backhand, and even volleys double-handed.

 He was Austria's most successful doubles player in history by reaching world no. 6 in the ATP doubles rankings in January 2008, before being matched by Jürgen
Melzer, who reached no. 6 in September 2010, and overtaken by Alexander Peya, who reached no. 3 in August 2013.
</item 1>

<item 2>
1983 Bristol Open – Doubles
Tim Gullikson and Tom Gullikson were the defending champions, but Tim Gullikson did not participate this year.

 Tom Gullikson partnered Johan Kriek, losing in the final.
</item 2>

<item 3>
Iveta Benešová
Iveta Benešová (] ) (formerly Melzer, Czech: Melzerová ; born 1 February 1983) is a Czech former professional tennis player.

 She began playing tennis at age of 7 and turned professional in 1998 in Prague.

 She has won two WTA Tour events and one Grand Slam in mixed doubles partnering with Jürgen Melzer at the 2011 Wimbledon Championships.

 On 14 September 2012 she married Jürgen Melzer and adopted his family name.

 She announced her retirement from professional tennis via her Facebook page 13/08/2014.

 In 2015, she divorced Melzer and reverted to using her maiden name.
</item 3>

<item 4>
2012 US Open – Men's Doubles
Jürgen Melzer and Philipp Petzschner were the defending champions, but lost in the second round to Jamie Delgado and Ken Skupski.

 Melzer and Petzschner needed a wildcard to compete because they forgot to sign up.

<br>
</item 4>

<item 5>
Tom Gullikson
Tom Gullikson (born September 8, 1951) is a tennis coach and former professional tennis player from the United States.
</item 5>

<item 6>
Jürgen Melzer
Jürgen Melzer (born 22 May 1981 in Vienna) is an Austrian tennis player.

 He reached a career-high singles ranking of world No. 8 in April 2011, and a doubles ranking of world No. 6 in September 2010.

 He is a left-handed tennis player, but is right-handed in everyday life.

 He has a younger brother, Gerald Melzer, with whom he has played doubles in several tournaments.
</item 6>

<item 7>
2010 Pilot Pen Tennis – Men's Doubles
Julian Knowle and Jürgen Melzer were the defending champions, but Melzer chose not to participate this year.As a result, Knowle partnered with Andy Ram, but
they lost to Robert Lindstedt and Horia Tecău in the semifinals.

 Lindstedt and Tecău went on to win the tournament, after defeating Rohan Bopanna and Aisam-ul-Haq Qureshi in the final 6–4, 7–5.
</item 7>

<item 8>
Gerald Melzer
Gerald Melzer (born 13 July 1990) is a professional Austrian tennis player.

 As a qualifier, he reached the semifinals of Münich in 2015.

 He achieved a career-high singles ranking of world No. 68 in November 2016.
</item 8>

<item 9>
1983 Wimbledon Championships – Men's Doubles
The Men's Doubles tournament at the 1983 Wimbledon Championships was held from August 30 to September 11, 1983, on the outdoor grass courts at the All England
Lawn Tennis and Croquet Club in the Wimbledon district of London, England.

 Peter McNamara and Paul McNamee were the defending champions.

 Peter McNamara did not compete this year while McNamee partnered with Brian Gottfried, losing in the quarterfinals to Anders Jarryd and Hans Simonsson.

 Peter Fleming and John McEnroe won the title, defeating Tim Gullikson and Tom Gullikson in the final.
</item 9>

<item 10>
Pam Teeguarden
Teeguarden was the first woman tennis player signed by Nike.

 She played on the victorious Los Angeles Strings Team Tennis team in 1981 and won the Team Tennis Mixed Doubles Division with Tom Gullikson in 1977; they were
also runners-up in the league that year.
</item 10>
=== END ITEM id=9ce0ca89d16b8230 ===

=== ITEM id=2739ec9fe8ea8493 ===
Gold sentences and facts:
Sentence 0: Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
  Fact 0.0: Ruggero Deodato was born on 7 May 1939.
  Fact 0.1: Ruggero Deodato is an Italian film director.
  Fact 0.2: Ruggero Deodato is an Italian screenwriter.
  Fact 0.3: Ruggero Deodato is an Italian actor.
Sentence 1: Alejandro Springall is a Mexican film director and producer.
  Fact 1.0: Alejandro Springall is a Mexican film director.
  Fact 1.1: Alejandro Springall is a Mexican producer.

Retrieved items:
<item 1>
Alejandro Springall
Alejandro Springall is a Mexican film director and producer.
</item 1>

<item 2>
Victor Salva
Victor Ronald Salva (born March 29, 1958) is an American film director and screenwriter.

 He directed the films "Powder" (1995) and "Jeepers Creepers" (2001).

 The latter of these has been expanded into a franchise with two sequels that he has also directed.
</item 2>

<item 3>
Antonio De Carlo
Antonio De Carlo, born (1967--) 4, 1967 (age 50 ) in Tijuana, Baja California, Mexico, is a Mexican actor also known as Antonio in the late 1980s and early
1990s.

 He won an Emmy Award in 2005-2006.

 He is president and founder of the "Fundación Cultura Sin Fronteras AC".

 After twelve years out of show business, in 2012 he returned to the soap operas performing "Magic Dragon" one of the main characters for "Miss XV", a
co-production from Televisa México and Nickelodeon directed and produced by Pedro Damian.

 He is also part of the cast of "Muchacha Italiana Viene a Casarse", directed by Pedro Damian (Televisa, México).
</item 3>

<item 4>
John Steiner
He found himself in demand in Italy and moved there, appearing in a great number of Italian exploitation and B-films including police actioners ("Violent
Rome"), westerns ("Mannaja"), war films ("The Last Hunter"), nazisploitation ("Deported Women of the SS Special Section"), sci-fi adventure films ("Yor, the
Hunter from the Future"), and horror films, such as Mario Bava's "Shock", Dario Argento's "Tenebrae", and Ruggero Deodato's "Body Count".

 He also became a favourite of famed Italian filmmaker Tinto Brass, featuring in "Salon Kitty", the infamous "Caligula", "Action", and "Paprika".
</item 4>

<item 5>
Ruggero Deodato
Ruggero Deodato (born 7 May 1939) is an Italian film director, screenwriter, and actor.
</item 5>

<item 6>
Cannibal Holocaust
Cannibal Holocaust is a 1980 Italian cannibal horror film directed by Ruggero Deodato from a screenplay by Gianfranco Clerici.

 It stars Carl Gabriel Yorke, Robert Kerman, Francesca Ciardi, Perry Pirkanen, and Luca Barbareschi.

 Influenced by the works of Mondo director Gualtiero Jacopetti, the film was inspired by Italian media reporting of Red Brigade terrorism.

 The coverage included news reports Deodato believed to be staged, an idea which became an integral aspect of the film's story.

 "Cannibal Holocaust" was filmed primarily in the Amazon rainforest of Colombia with indigenous tribes interacting with American and Italian actors.
</item 6>

<item 7>
Tinto Brass
Giovanni "Tinto" Brass (born 26 March 1933) is an Italian filmmaker.

 In the 1960s and 1970s, he directed many critically acclaimed avant-garde films of various genres.

 Today, he is mainly known for his later work in the erotic genre, with films such as "Caligula", "Così fan tutte" (released under the English title "All Ladies
Do It"), "Paprika", "Monella" ("Frivolous Lola") and "Trasgredire".
</item 7>

<item 8>
Graph query row:
person: Alejandro Springall
nationality: None
occupation: director
is_film_director: True
</item 8>

<item 9>
Graph query row:
person: Ruggero Deodato
nationality: Italian
occupation: film director
is_film_director: True
</item 9>

<item 10>
Hercules, Prisoner of Evil
Hercules, Prisoner of Evil (Italian: "Ursus il terrore dei Kirghisi" , 'Ursus, Terror of the Kirghiz' ) is a 1964 Italian "peplum" film directed by Antonio
Margheriti and an uncredited Ruggero Deodato.

 Deodato, the official assistant director, replaced Margheriti as he was busy with the completion of the film "The Fall of Rome".

 Deodato actually directed most of the film in actuality but Margheriti was credited as the director.

 The film is filled with a variety of horrific themes and elements, featuring a killer werewolf, and is as much a horror film as it is a peplum.
</item 10>
=== END ITEM id=2739ec9fe8ea8493 ===

=== ITEM id=f951da310925102d ===
Gold sentences and facts:
Sentence 0: Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space as part of the Soviet Union's Interkosmos programme.
  Fact 0.0: Sigmund Werner Paul Jähn was born on 13 February 1937.
  Fact 0.1: Sigmund Werner Paul Jähn is a German cosmonaut.
  Fact 0.2: Sigmund Werner Paul Jähn is a pilot.
  Fact 0.3: In 1978 Sigmund Werner Paul Jähn became the first East German to fly in space.
  Fact 0.4: In 1978 Sigmund Werner Paul Jähn became the first German native to fly in space.
  Fact 0.5: The 1978 flight by Sigmund Werner Paul Jähn was part of the Soviet Union's Interkosmos programme.
Sentence 1: Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.
  Fact 1.0: Arne Christer Fuglesang was born on March 18, 1957.
  Fact 1.1: Arne Christer Fuglesang was born in Stockholm.
  Fact 1.2: Arne Christer Fuglesang is a Swedish physicist.
  Fact 1.3: Arne Christer Fuglesang is an ESA astronaut.
Sentence 2:  He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
  Fact 2.0: Arne Christer Fuglesang was first launched aboard the STS-116 Space Shuttle mission.
  Fact 2.1: That launch was on December 10, 2006, at 01:47 GMT.
  Fact 2.2: Arne Christer Fuglesang became the first Swedish citizen in space.

Retrieved items:
<item 1>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 1>

<item 2>
The profiles cover NASA computer scientist Annie Easley and space espresso: ISSpresso’s first in-space coffee, brewed by Samantha Cristoforetti in 2015. They
describe astrophysicist Jean-Loup Puget; missions Soyuz TM-7/T-6 and French astronaut Jean-Loup Chrétien; Patrick Baudry; Project Highwater; Apollo 7;
SpaceShipOne’s first privately funded human spaceflight; the Soyuz rocket; Lockheed test pilot John Christiansen
</item 2>

<item 3>
Hans Schlegel
Hans Wilhelm Schlegel (Überlingen, 3 August 1951) is a German physicist, an ESA astronaut, and a veteran of two NASA Space Shuttle missions.
</item 3>

<item 4>
The entries cover German rocket scientist Magnus “Mac” von Braun, Wernher’s brother; Wernher von Braun’s Ferry Rocket shuttle concept and the 1937 failed
Aggregate 3 launches (Operation Lighthouse); and Collier’s influential 1950s spaceflight series *Man Will Conquer Space Soon!*. They also profile German
astronauts Hans Schlegel, Ernst Messerschmid, Ulrich Walter, Sigmund Jähn, Ulf Merbold, and Soyuz
</item 4>

<item 5>
Claude Nicollier
Claude Nicollier (born 2 September 1944 in Vevey, Switzerland) is the first astronaut from Switzerland.

 He has flown on four Space Shuttle missions.

 His first spaceflight (STS-46) was in 1992, and his final spaceflight (STS-103) was in 1999.

 He took part in two servicing missions to the Hubble Space Telescope (called STS-61 and STS-103).

 During his final spaceflight he participated in a spacewalk, becoming the first European Space Agency astronaut to do so during a Space Shuttle mission
(previous ESA astronauts conducted spacewalks aboard "Mir", see List of spacewalks and moonwalks 1965–1999).

 In 2000 he was assigned to the Astronaut Office Extravehicular Activity Branch, while maintaining a position as Lead ESA Astronaut in Houston.

 Nicollier retired from ESA in April 2007.
</item 5>

<item 6>
Christer Fuglesang
Arne Christer Fuglesang (] ) (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut.

 He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, at 01:47 GMT, making him the first Swedish citizen in space.
</item 6>

<item 7>
Ulf Merbold
Dr. Ulf Dietrich Merbold (born June 20, 1941) is the first West German citizen and second German native (after Sigmund Jähn) to have flown in space.

 He is also the first member of the European Space Agency Astronaut Corps to participate in a spaceflight mission and the first non-US citizen to reach orbit in
a US spacecraft.

 In 1983, he and Byron Lichtenberg became the first Payload Specialists to fly on the shuttle.
</item 7>

<item 8>
Sigmund Jähn
Sigmund Werner Paul Jähn (born 13 February 1937) is a German cosmonaut and pilot, who in 1978 became the first East German (and German native) to fly in space
as part of the Soviet Union's Interkosmos programme.
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
Eberhard Köllner
Eberhard Köllner (born 29 September 1939 in Stassfurt, Germany) was selected for Soyuz 31 as the backup for Sigmund Jähn.
</item 10>
=== END ITEM id=f951da310925102d ===

=== ITEM id=40782df085ba4fc7 ===
Gold sentences and facts:
Sentence 0:  It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
  Fact 0.0: The series was created by Jay Lender.
  Fact 0.1: The series was created by Micah Wright.
  Fact 0.2: Jay Lender was a former writer for The Angry Beavers.
  Fact 0.3: Jay Lender was a former writer for SpongeBob SquarePants.
  Fact 0.4: Jay Lender was a former writer for Phineas and Ferb.
  Fact 0.5: Micah Wright was a former writer for The Angry Beavers.
  Fact 0.6: Micah Wright was a former writer for SpongeBob SquarePants.
  Fact 0.7: Micah Wright was a former writer for Phineas and Ferb.
Sentence 1:  The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.
  Fact 1.0: The series chronicles the adventures and endeavors of the title character and his various friends.
  Fact 1.1: The series is set in the fictional underwater city of Bikini Bottom.

Retrieved items:
<item 1>
SpongeBob SquarePants: Battle for Bikini Bottom
SpongeBob SquarePants: Battle for Bikini Bottom is a 2003 video game based on the animated series of the same name, developed by Heavy Iron Studios, AWE Games,
and Vicarious Visions, and published by THQ.

 The game was released for the PlayStation 2, Game Boy Advance (GBA), Xbox, and Nintendo GameCube consoles as well as a port for Microsoft Windows.

 The Microsoft Windows version of the game is a point-and-click adventure game, whereas the 6th generation platforms received a 3D platformer.

 The Game Boy Advance version received a 2D platformer.

 All versions of the game feature an original storyline, in which the player attempts to defend Bikini Bottom from an invasion of robots created by Plankton
with a machine called the Duplicatotron 3000, playing as SpongeBob in all versions, as well as Patrick and Sandy in the console versions.

 The game was released on October 31, 2003, in North America and in Europe on November 28, 2003.
</item 1>

<item 2>
Robocalypse
Robocalypse is a real-time strategy video game developed and published by Vogster Entertainment for the Nintendo DS handheld video game console.

 It has only been released in North America.

 It was created by Jay Lender and Micah Wright, former writers for "The Angry Beavers", "SpongeBob SquarePants", and "Phineas and Ferb".
</item 2>

<item 3>
Adventure Time
Adventure Time is an American animated television series created by Pendleton Ward for Cartoon Network.

 It follows the adventures of a boy named Finn (voiced by Jeremy Shada) and his best friend and adoptive brother Jake (John DiMaggio)— a dog with the magical
power to change shape and size at will.

 Finn and Jake live in the post-apocalyptic Land of Ooo, where they interact with Princess Bubblegum (Hynden Walch), the Ice King (Tom Kenny), Marceline the
Vampire Queen (Olivia Olson), BMO (Niki Yang), and others.

 The series is based on a 2007 short produced for Nicktoons and Frederator Studios' animation incubator series "Random!

 Cartoons".

 After the short became a viral hit on the Internet, Cartoon Network commissioned a full-length series, which previewed on March 11, 2010, and officially
premiered on April 5, 2010.
</item 3>

<item 4>
Rock Bottom (SpongeBob SquarePants)
"Rock Bottom" is the 34th episode of the first season of the American animated television series "SpongeBob SquarePants".

 It originally was produced in 1999 and aired on Nickelodeon in the United States on March 15, 2000.

 The series follows the adventures of the title character in the underwater city of Bikini Bottom.

 In the episode, SpongeBob becomes stranded in a abyssopelagic zone called Rock Bottom.
</item 4>

<item 5>
SpongeBob SquarePants (season 3)
The third season of the American animated television series "SpongeBob SquarePants", created by Stephen Hillenburg, aired on Nickelodeon from October 5, 2001 to
October 11, 2004, and consists of 20 episodes.

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, who also acted as the showrunner.

 Hillenburg halted production on the show to work on the 2004 film adaptation of the series, "The SpongeBob SquarePants Movie".

 After production on the film, Hillenburg resigned from the show as its showrunner, and appointed staff writer Paul Tibbitt to overtake the position.

 Season 3 was originally set to end the series after the release of the film, but the success prevented the series from ending, leading to a fourth season.
</item 5>

<item 6>
Band Geeks
"Band Geeks" is the second part of the 15th episode of the second season, and the second half of the 35th episode overall, of the American animated television
program "SpongeBob SquarePants".

 It originally aired on Nickelodeon in the United States on September 7, 2001.

 It was written by C. H. Greenblatt, Aaron Springer, and Merriwether Williams, and the animation was directed by Frank Weiss.

 Springer served as storyboard director, and Greenblatt served as storyboard artist.

 The song "Sweet Victory" by David Glen Eisley was featured in the episode and was later released on the album "" in 2005.
</item 6>

<item 7>
List of SpongeBob SquarePants episodes
"SpongeBob SquarePants" is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series is set in the fictional underwater city of Bikini Bottom, and centers on the adventures and endeavors of SpongeBob SquarePants, an over-optimistic
sea sponge that annoys other characters.

 Many of the ideas for the show originated in an unpublished, educational comic book titled "The Intertidal Zone", which Hillenburg created in the mid-1980s.

 He began developing "SpongeBob SquarePants" into a television series in 1996 after the cancellation of "Rocko's Modern Life", another Nickelodeon television
series which Hillenburg previously directed.
</item 7>

<item 8>
SpongeBob SquarePants (season 4)
The fourth season of the American animated television series "SpongeBob SquarePants", created by former marine biologist and animator Stephen Hillenburg, aired
on Nickelodeon from May 6, 2005 to July 24, 2007, and contained 20 episodes, beginning with the episodes "Fear of a Krabby Patty" and "Shell of a Man".

 The series chronicles the exploits and adventures of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The season was executive produced by series creator Hillenburg, while writer Paul Tibbitt acted as the supervising/co-executive producer and showrunner.

 The show underwent a hiatus on television as Hillenburg halted the production in 2002 to work on the film adaptation of the series, "The SpongeBob SquarePants
Movie".

 Once the film was finalized and the previous season had completed broadcast on television, Hillenburg wanted to end the show, but the success of the series led
to more episodes, so Tibbitt took over Hillenburg's position as showrunner and began working on a fourth season for broadcast in 2005.
</item 8>

<item 9>
SpongeBob SquarePants
SpongeBob SquarePants is an American animated television series created by marine biologist and animator Stephen Hillenburg for Nickelodeon.

 The series chronicles the adventures and endeavors of the title character and his various friends in the fictional underwater city of Bikini Bottom.

 The series' popularity has made it a media franchise, as well as the highest rated series to ever air on Nickelodeon, and the most distributed property of MTV
Networks.

 As of 2015, the media franchise has generated $12 billion in merchandising revenue for Nickelodeon.
</item 9>

<item 10>
Legends of Bikini Bottom
Legends of Bikini Bottom is an anthology series of six episodes in the American animated television series "SpongeBob SquarePants", as part of its seventh
season.

 As the name suggests, the episodes have plots involving things like monsters, magic and mythical creatures.

 The first episode called "Trenchbillies" first premiered on the online social networking service Facebook on January 27, 2011 before airing on Nickelodeon on
January 29, 2011.

 The four other episodes titled "Sponge-Cano!"

, "The Main Drain", "The Monster Who Came to Bikini Bottom" and "Welcome to the Bikini Bottom Triangle" premiered on Nickelodeon as part of an hour-long special
on January 28, 2011.

 The decision to air the series online was aimed at attracting "the young and the restless hooked to the internet and the social media."
</item 10>
=== END ITEM id=40782df085ba4fc7 ===

=== ITEM id=4b9d8e9c6ce6d116 ===
Gold sentences and facts:
Sentence 0:  They have produced two studio albums and an EP.
  Fact 0.0: They have produced two studio albums.
  Fact 0.1: They have produced an EP.
Sentence 1: Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.
  Fact 1.0: Motion City Soundtrack was an American rock band.
  Fact 1.1: Motion City Soundtrack formed in Minneapolis, Minnesota.
  Fact 1.2: Motion City Soundtrack formed in 1997.
Sentence 2:  Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.
  Fact 2.0: Motion City Soundtrack had a nearly twenty-year career.
  Fact 2.1: Motion City Soundtrack toured heavily over the course of its career.
  Fact 2.2: Motion City Soundtrack released six studio albums.
  Fact 2.3: The majority of Motion City Soundtrack's six studio albums were released on the independent label Epitaph Records.

Retrieved items:
<item 1>
Panic Stations (album)
Panic Stations is the sixth and final studio album by American rock band Motion City Soundtrack.

 Produced by John Agnello, the album was released on September 18, 2015 by Epitaph Records.

 Following an extended touring cycle in which the group performed several past albums in full, Motion City Soundtrack lost one of its longest-serving members:
drummer Tony Thaxton, who resigned from the group in 2013.

 The band continued to tour with new drummer Claudio Rivera over the ensuing year while writing new material for a sixth album.
</item 1>

<item 2>
My Dinosaur Life
My Dinosaur Life is the fourth studio album by the American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on January 19, 2010 by Columbia Records.

 After many years on independent label Epitaph Records, Motion City Soundtrack signed to major label Columbia in 2006, prior to the release of their third
album, "Even If It Kills Me" (2007).

 Blink-182 bassist Mark Hoppus returned to produce the album; he had previously worked with the band on their 2005 breakthrough "Commit This to Memory".
</item 2>

<item 3>
Motion City Soundtrack discography
The following is the discography of Motion City Soundtrack, an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 3>

<item 4>
Commit This to Memory
Commit This to Memory is the second studio album by American rock band Motion City Soundtrack.

 Produced by Mark Hoppus, the album was released on June 7, 2005, in the United States by Epitaph Records.

 Motion City Soundtrack, formed in 1997, had first found success with their debut album, "I Am the Movie" (2002).

 The band toured in the interim years, creating positive word-of-mouth.

 In 2004, the band joined Blink-182 on the road for a string of shows, which led to their bassist, Mark Hoppus, joining the band in the studio for his first
producing effort.
</item 4>

<item 5>
Even If It Kills Me
Even if It Kills Me is the third studio album by American rock band Motion City Soundtrack.

 Produced by Ric Ocasek, Adam Schlesinger and Eli Janney, the album was released on September 18, 2007, in the United States by Epitaph Records.

 Motion City Soundtrack, based in Minneapolis, Minnesota, made a breakthrough with their second album, "Commit This to Memory", garnering praise and independent
buzz upon its 2005 release.

 Following the release, the band toured relentlessly over the next two years, during which time frontman Justin Pierre struggled with alcohol and substance
abuse.

 These addictions were infused into the writing process of "Even If Kills Me", which was recorded in early 2007 Stratosphere Studios in Chelsea and the
legendary Electric Lady Studios in Greenwich Village.
</item 5>

<item 6>
Back to the Beat (EP)
Back to the Beat was the first significant EP release by the Minneapolis based pop punk band Motion City Soundtrack.

 The EP was recorded in Sound In Motion Recordings and released by Modern Radio Records and includes tracks that were later re-released.

 "Throw Down", "Back to the Beat" and "Capital H" were re-released on a split EP with Schatzi.

 "Capital H" was re-released on Motion City Soundtrack's first album "I Am the Movie".

 At the time of recording, the band members differed from the Motion City Soundtrack line-up formed in 2002.

 They were Joshua Cain on guitar and vocals, Austin Lindstrom on bass guitar, Joel Habedank on drums and Justin Pierre on lead vocals, guitar and keyboard.
</item 6>

<item 7>
Motion City Soundtrack
Motion City Soundtrack was an American rock band that formed in Minneapolis, Minnesota in 1997.

 The band's line-up consisted of vocalist and guitarist Justin Pierre, lead guitarist Joshua Cain, keyboardist Jesse Johnson, bassist Matthew Taylor, and
drummer Tony Thaxton.

 Over the course of their nearly twenty-year career, the group toured heavily and released six studio albums, the majority on independent label Epitaph Records.

 The band's sound, at times described as pop punk or emo, made notable use of the Moog synthesizer.
</item 7>

<item 8>
Motion City Soundtrack, a Minneapolis rock band formed in 1997, released six studio albums over nearly 20 years, blending pop punk/emo with prominent Moog
synthesizers. Justin Pierre wrote most of the band’s 107 recorded songs, including nine covers. Key releases include *Commit This to Memory* (2005), produced by
Mark Hoppus; *Even If It Kills Me* (2007), shaped by Pierre’s struggles with addiction; *My Dinosaur Life* (
</item 8>

<item 9>
Candlelight Red
Candlelight Red is a rock band from Williamsport, Pennsylvania.

 They have produced two studio albums and an EP.

 Their EP "Demons" and album "Reclamation" was produced by Morgan Rose of Sevendust.
</item 9>

<item 10>
List of songs recorded by Motion City Soundtrack
American rock band Motion City Soundtrack recorded songs for six studio albums, various soundtracks, compilations, and non-album singles.

 The majority of the group's original material was written by vocalist and guitarist Justin Pierre.

 The band also recorded cover versions of other artists' songs at various points throughout their career, including by the Beatles, Limbeck, the Police, R.E.M.,
the Rentals, Rilo Kiley, Trampled by Turtles, and the Weakerthans.

 In all, the group recorded 107 songs, nine of which were covers.
</item 10>
=== END ITEM id=4b9d8e9c6ce6d116 ===

=== ITEM id=66df349735c4316d ===
Gold sentences and facts:
Sentence 0:  His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed according to its experimental approach.
  Fact 0.0: His experimental works are well known in the Malayalam Cinema industry.
  Fact 0.1: His experimental works are appropriately discussed in the Malayalam Cinema industry.
  Fact 0.2: His debut movie is Jalachhayam.
  Fact 0.3: Jalachhayam was released in 2010.
  Fact 0.4: Jalachhayam was well discussed according to its experimental approach.
Sentence 1: Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA, Britain and Germany.
  Fact 1.0: Jacques Feyder was born on 21 July 1885.
  Fact 1.1: Jacques Feyder died on 24 May 1948.
  Fact 1.2: Jacques Feyder was a Belgian actor.
  Fact 1.3: Jacques Feyder was a Belgian screenwriter.
  Fact 1.4: Jacques Feyder was a Belgian film director.
  Fact 1.5: Jacques Feyder worked principally in France.
  Fact 1.6: Jacques Feyder also worked in the USA.
  Fact 1.7: Jacques Feyder also worked in Britain.
  Fact 1.8: Jacques Feyder also worked in Germany.

Retrieved items:
<item 1>
Michel Foucault
Paul-Michel Foucault (15 October 1926 – 25 June 1984), generally known as Michel Foucault (] ), was a French philosopher, historian of ideas, social theorist,
and literary critic.
</item 1>

<item 2>
Shafiq Syed
Shafiq Syed (born 1976) is a former Indian child actor turned auto rickshaw driver.

 At the age of twelve, he starred in a lead role in the Mira Nair's Academy Award-nominated film "Salaam Bombay!

" (1988), for which he also received the 1989 National Film Award for Best Child Artist at the 36th National Film Awards.

 He drives an auto rickshaw in Bangalore and also works as an assistant in television production units.
</item 2>

<item 3>
Jacques Feyder
Jacques Feyder (] ; 21 July 1885 – 24 May 1948) was a Belgian actor, screenwriter and film director who worked principally in France, but also in the USA,
Britain and Germany.

 He was a leading director of silent films during the 1920s, and in the 1930s he became associated with the style of poetic realism in French cinema.

 He adopted French nationality in 1928.
</item 3>

<item 4>
Jacques Natanson
French writer Jacques Natanson (15 May 1901 – 19 May 1975) first became involved in the movies in 1929 when one of his plays was adapted for the screen.

 He enjoyed a fruitful collaboration with Max Ophüls, on such films as "La Ronde" (1951, earning an Academy Award nomination), "Le Plaisir" (1952) and "Lola
Montès" (1955).
</item 4>

<item 5>
Patrick Baudry
Patrick Pierre Roger Baudry (born March 6, 1946 in Cameroon), is a retired Lieutenant Colonel in the French Air Force and a former CNES astronaut.

 In 1985, he became the second French citizen in space, after Jean-Loup Chrétien, when he flew aboard NASA's Space Shuttle mission STS-51-G.
</item 5>

<item 6>
Sathish Kalathil
Sathish Kalathil (മലയാളം: ) is an Indian film and documentary Director and Producer in malayalam.

 He is also Story Writer, and Lyricist.

 His experimental works are well known and appropriately discussed in Malayalam Cinema industry and his debut movie Jalachhayam (2010) was well discussed
according to its experimental approach.

 Actually, his first venture was in 2008 named Veena Vaadanam, a documentary film about Art (Painting) and that was his first experimental entry in movie.

 In 2012, he directed Laloorinu Parayanullathu, a social committed documentary film about Municipal solid waste.
</item 6>

<item 7>
Alain Jessua
Alain Jessua (born 16 January 1932) is a French film director and screenwriter.

 He directed ten films between 1956 and 1997.

 He worked as assistant director for Jacques Becker on the set of "Casque d'or", with Max Ophüls for "Madame de..." and "Lola Montès" and with Marcel Carné on
"Wasteland". "

Léon la lune" his first short film won the influential Prix Jean Vigo in 1957.

 He directed first feature film in 1963 "La vie à l'envers" that won Best First Film at Venice Film Festival, in 1964.
</item 7>

<item 8>
Bertalan Farkas
Bertalan Farkas (born August 2, 1949) is the first Hungarian cosmonaut and the first Esperantist in space.

 He is currently the president of Airlines Service and Trade.

 With Charles Simonyi's travel, Farkas is no longer the only Hungarian who has been to space (he is still the only astronaut, as Simonyi flew as a space
tourist).
</item 8>

<item 9>
Kevin Mayer
Kevin Mayer (] , ] or ] , born 10 February 1992) is a French athlete specializing in combined events.

 A native of Drôme in southeast France, Mayer excelled in combined events from a young age; highlights from his youth career include winning gold medals in the
octathlon at the 2009 World Youth Championships and in the decathlon at the 2010 World Junior Championships and at the 2011 European Athletics Junior
Championships, establishing a new national junior record at the latter.
</item 9>

<item 10>
Émile Verdet
Marcel Émile Verdet (13 March 1824 – 3 June 1866) was a French physicist.

 He worked in magnetism and optics, editing the works of Augustin-Jean Fresnel.

 Verdet did much to champion the early theory of the conservation of energy in France through his editorial supervision of the "Annales de chimie et de
physique".
</item 10>
=== END ITEM id=66df349735c4316d ===
