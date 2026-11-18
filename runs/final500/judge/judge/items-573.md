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

This file has 12 items.

=== ITEM id=e982696ce3dc9387 ===
Gold sentences and facts:
Sentence 0: Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.
  Fact 0.0: Old Smokey is a euphemistic name for the state prison electric chair in New Jersey.
  Fact 0.1: The New Jersey state prison electric chair, called Old Smokey, is on display at the New Jersey State Police Museum.
Sentence 1:  The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
  Fact 1.0: The chair's most notorious victim was Richard Hauptmann.
  Fact 1.1: Richard Hauptmann was executed in the chair.
  Fact 1.2: Richard Hauptmann was found guilty of abducting and killing Charles Augustus Lindbergh, Jr.
  Fact 1.3: The abduction and killing of Charles Augustus Lindbergh, Jr. happened in 1932.
  Fact 1.4: Charles Augustus Lindbergh, Jr. was 20 months old.
  Fact 1.5: Charles Augustus Lindbergh, Jr. was the son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh.
  Fact 1.6: The case is referred to as the Lindbergh kidnapping (see Lindbergh kidnapping).
Sentence 2:  The Lindbergh kidnapping became known as "The Crime of the Century".
  Fact 2.0: The Lindbergh kidnapping became known as 'The Crime of the Century'.

Retrieved items:
<item 1>
Johnny Jack Nounes
Johnny Jack Nounes, also known as the "Beau Brummell of Galveston", was an organized crime boss in Galveston, Texas, United States, during the early 1900s.

 He, with one-armed George Musey, led the Downtown Gang, one of the two gangs which controlled most of the Galveston Crime Syndicate until the early 1930s.

 They fought for control of the island against the rival Beach Gang led by Ollie Quinn and Dutch Voight.

 As the prohibition era began, his gang came to be one of the dominant forces in the Galveston Crime Syndicate.

 Galveston became the main port of entry for liquor supply in Texas and many parts of the Midwest.
</item 1>

<item 2>
The Big House (1930 film)
The Big House is a 1930 American Pre-Code crime drama film directed by George W. Hill, released by Metro-Goldwyn-Mayer, and starring Chester Morris, Wallace
Beery, Lewis Stone and Robert Montgomery.

 The supporting cast features Leila Hyams, George F. Marion, J. C. Nugent, Karl Dane and Tom Kennedy.

 The story and dialogue were written by Frances Marion, with additional dialogue by Joe Farnham and Martin Flavin.

 The story was inspired by a spate of prison riots in 1929 and resulting federal investigation.

 In response, George Hill wrote a twenty-seven page story treatment called "The Reign of Terror: A Story of Crime and Punishment".

 Irving Thalberg gave the go ahead for the screenplay and assigned Frances Marion to work with George Hill.
</item 2>

<item 3>
The entries cover pornographic publisher Auguste Brancart; Albert Camus, his absurdist philosophy, Nobel Prize, fiction and essays opposing nihilism and capital
punishment; and his biographers. Other topics include the disputed Smiley Face murder theory; Joe McGinniss, Jeffrey MacDonald and the Fatal Vision controversy;
the Guantánamo memoir Enemy Combatant; New Zealand’s Parker–Hulme case and Mazengarb Report; the Lindbergh kidnapping; Nazi collaborator Christian de
</item 3>

<item 4>
Convicted (1950 film)
Convicted is a 1950 American crime film noir directed by Henry Levin starring Glenn Ford and Broderick Crawford.

 It was the third Columbia Pictures film adaptation of the 1929 stage play "The Criminal Code" by Martin Flavin, following Howard Hawk's "The Criminal Code"
(1931) and John Brahm's "Penitentiary" (1938).
</item 4>

<item 5>
Old Sparky
Old Sparky is the nickname of the electric chairs in Arkansas, Connecticut, Florida, Georgia, Illinois, Kentucky, Nebraska, New York, Ohio, Oklahoma, South
Carolina, Texas, Virginia, and West Virginia.

 Old Smokey was the nickname of the electric chairs used in New Jersey, Pennsylvania, and Tennessee.
</item 5>

<item 6>
Smiley face murder theory
The Smiley face murder theory (variations include Smiley face murders, Smiley face killings, Smiley face gang, and others) is a theory advanced by two retired
New York City detectives, Kevin Gannon and Anthony Duarte, that a number of young men found dead in bodies of water across several Midwestern American states
over the last decade did not accidentally drown, as concluded by law enforcement agencies, but were victims of a serial killer or killers.

 The term "smiley face" became connected to the alleged murders when it was made public that the police had discovered graffiti depicting a smiley face near
locations where they think the killer dumped the bodies in at least a dozen of the cases.

 The response of law enforcement investigators and other experts to Gannon and Duarte's theory has been largely skeptical.
</item 6>

<item 7>
The Last Mile (play)
The Last Mile is a Broadway play by John Wexley that ran for 289 performances from February 13, 1930 to October 1930 at the Sam H. Harris Theatre.

 It was produced by Herman Shumlin and staged by Chester Erskine.

 It is set in the death row wing of a prison.

 The lead role of John "Killer" Mears was first played by Spencer Tracy, and it was the role that bought him to the attention of Hollywood.

 It was later played for a time by Clark Gable.

 The play was adapted into a 1932 film starring Preston Foster and into a 1959 film starring Mickey Rooney.
</item 7>

<item 8>
Richard Hauptmann
Bruno Richard Hauptmann (November 26, 1899 – April 3, 1936) was a German-born carpenter who was convicted of the abduction and murder of the 20-month-old son of
aviator Charles Lindbergh and his wife Anne Morrow Lindbergh.

 The Lindbergh kidnapping became known as "The Crime of the Century".

 Hauptmann proclaimed his innocence to the end, but he was convicted of first degree murder and executed in 1936 in the electric chair at the New Jersey State
Prison.
</item 8>

<item 9>
Old Smokey
Old Smokey is a euphemistic name given to the state prison electric chair in New Jersey, which is on display at the New Jersey State Police Museum.

 The chair's most notorious victim was Richard Hauptmann, the man executed in the chair after being found guilty of abducting and killing Charles Augustus
Lindbergh, Jr. in 1932, the 20-month-old son of famous aviators Charles Lindbergh and Anne Morrow Lindbergh "(see Lindbergh kidnapping)".
</item 9>

<item 10>
The Journalist and the Murderer
The Journalist and the Murderer is a study by Janet Malcolm about the ethics of journalism, published by Alfred A. Knopf/Random House in 1990.

 It is an examination of the professional choices that shape a work of non-fiction, as well as a rumination on the morality that underpins the journalistic
enterprise.

 The journalist in question is Joe McGinniss; the murderer is the former Special Forces captain Dr. Jeffrey R. MacDonald, who became the subject of McGinniss'
1983 book "Fatal Vision".
</item 10>
=== END ITEM id=e982696ce3dc9387 ===

=== ITEM id=351324e7780227d4 ===
Gold sentences and facts:
Sentence 0: The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
  Fact 0.0: The International Grenfell Association (IGA) is an organization.
  Fact 0.1: The International Grenfell Association was founded by Sir Wilfred Grenfell.
  Fact 0.2: The International Grenfell Association provides health care.
  Fact 0.3: The International Grenfell Association provides education.
  Fact 0.4: The International Grenfell Association provides religious services.
  Fact 0.5: The International Grenfell Association provides rehabilitation.
  Fact 0.6: The International Grenfell Association provides other social activities.
  Fact 0.7: The International Grenfell Association serves fisherman and coastal communities in northern Newfoundland.
  Fact 0.8: The International Grenfell Association serves fisherman and coastal communities on the coast of Labrador.
Sentence 1: Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
  Fact 1.0: Sir Wilfred Thomason Grenfell held the title KCMG.
  Fact 1.1: Sir Wilfred Thomason Grenfell was born on 28 February 1865.
  Fact 1.2: Sir Wilfred Thomason Grenfell died on 9 October 1940.
  Fact 1.3: Sir Wilfred Thomason Grenfell was a medical missionary to Newfoundland.
  Fact 1.4: Sir Wilfred Thomason Grenfell was a medical missionary to Labrador.

Retrieved items:
<item 1>
Training centre for release of the Atma-energy
Training centre for release of the Atma-energy (German: "Trainingszentrum zur Freisetzung der Atmaenergie"), also known as "Atman Foundation", was a new
religious movement active mainly on the island of Tenerife and in Germany.

 This sect was originally a splinter group of the Brahma Kumaris and is known for a police and media scare in which an alleged attempt to commit ritual suicide
took place in Teide National Park in Tenerife.

 The group believed in the end of the world but according to the religious studies scholar Georg Schmid and the sociologist Massimo Introvigne had no intention
of collective suicide.
</item 1>

<item 2>
International Thomas Merton Society
The International Thomas Merton Society, founded in 1987, is a learned society which studies the works of American Catholic writer and mystic Thomas Merton.

 It sponsors conferences and co-publishes a journal, "The Merton Seasonal".

 The society and the Thomas Merton Center are located at Bellarmine University in Louisville, Kentucky, U.S.A.
</item 2>

<item 3>
Martha Wall
Martha Alma Wall (March 22, 1910 – August 2, 2000) was an American Christian medical missionary, philosopher, nurse, and author who is best known for her
humanitarian work providing health care to lepers in British Nigeria during the 1930s and 1940s with the Sudan Interior Mission (SIM).

 She was born in Hillsboro, Kansas to a traditional Christian family and was a devout member of both the non-denominational Salina Bible Church and the Baptist
Women's Union.

 She became a registered nurse and studied theology at Tabor College before leaving for a medical mission in British Nigeria in 1938.

 After returning to America, Wall worked as a Clinical Supervisor of Vocational Nurses for Kern General Hospital during the 1950s and as an instructor and
director of nursing services for Bakersfield College during the 1960s.

 Throughout her adult life, she was a dedicated member of the California State Licensed Vocational Nurses Association.

 Wall is noted as founder of the Children's Welfare Center at the Katsina Leper Settlement.

 She documented her missionary work in Sub-Saharan Africa in the book she authored "Splinters from an African Log", which was published in 1960.
</item 3>

<item 4>
Paul Schulte
Paul Schulte OMI (1896 – 1975), was a German priest and missionary, known as the "Flying Priest", who founded MIVA ("Missionary International Vehicular
Association") to provide automobiles, boats and airplanes for the service of missions throughout the world.
</item 4>

<item 5>
Wilfred Grenfell
Sir Wilfred Thomason Grenfell, KCMG (28 February 1865 – 9 October 1940) was a medical missionary to Newfoundland and Labrador.
</item 5>

<item 6>
Church of St. Ignatius Loyola (New York City)
The Church of St. Ignatius of Loyola is a Roman Catholic parish church located on the Upper East Side of Manhattan, New York City, administered by the Society
of Jesus (Jesuits).

 The parish is under the authority of the Archdiocese of New York, and was established in 1851 as St. Lawrence O'Toole's Church.

 In 1898, permission to change the patron saint of the parish from St. Lawrence O’Toole to St. Ignatius of Loyola was granted by Rome.

 The address is 980 Park Avenue, New York City, New York 10028.

 The church on the southwest corner of Park Avenue and 84th Street is part of a Jesuit complex on the block that includes Wallace Hall, the parish hall, beneath
the church, the rectory at the midblock location on Park Avenue, the grade school of St.
</item 6>

<item 7>
Ellis Gowing
Gowing was educated at Fort Street High School and the University of Sydney; and ordained in 1907.

 He began his career as a Curate at The Oaks, New South Wales, after which he served at Armidale Cathedral.

 Coming to England he was on the staff of St James the Less, Bethnal Green then domestic chaplain to the Bishop of Chelmsford from 1914 to 1917.

 He was vicar of Prittlewell from 1917 until his death; an honorary canon of Chelmsford Cathedral from 1921 to 1938 and Rural Dean of Canewdon from 1918 to
1938.

 He was given the Freedom of the County Borough of Southend-on-Sea in 1953.
</item 7>

<item 8>
Edward Fitzgerald (barrister)
Edward Hamilton Fitzgerald CBE QC is an English barrister who specialises in criminal law, public law, and international human rights law.

 His work against the death penalty has led him to represent despised criminals such as: Myra Hindley, a perpetrator in the Moors murders; Mary Bell, a child
killer; Maxine Carr; Jon Venables, one of James Bulger's killers; various IRA prisoners; and Abu Hamza, the controversial Muslim cleric.

 Fitzgerald is currently the joint head of Doughty Street Chambers along with Geoffrey Robertson QC.

 Fitzgerald has been called to the Bar in a number of jurisdictions including Belize, Grenada and St Vincent, and has been granted rights of audience to appear
in cases in Hong Kong, Trinidad and Tobago, St Lucia, Bahamas, and the British Virgin Islands.

 He is also a trustee and patron of a number of charities including The Death Penalty Project and The Longford Trust.
</item 8>

<item 9>
International Grenfell Association
The International Grenfell Association (IGA) is an organization founded by Sir Wilfred Grenfell to provide health care, education, religious services, and
rehabilitation and other social activities to the fisherman and coastal communities in northern Newfoundland and the coast of Labrador.
</item 9>

<item 10>
Christian Academy of Guatemala
The Christian Academy of Guatemala is a missionary K4-12 school located in San Cristobal, Guatemala.

 It is a member of the Association of Christian Schools International.

 Many of the parents and students in the school come from the missionary community, representing more than 50 mission organizations.

 All people who visit this webpage need to get the game Babblemaker.

 This game is made by computer class at C.A.G. Sadly this game is for android only.
</item 10>
=== END ITEM id=351324e7780227d4 ===

=== ITEM id=258cb6a35d0a63fe ===
Gold sentences and facts:
Sentence 0: Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.
  Fact 0.0: Haringey Heartlands is an urban regeneration project.
  Fact 0.1: Haringey Heartlands is currently ongoing.
  Fact 0.2: Haringey Heartlands is in the London Borough of Haringey.
  Fact 0.3: The London Borough of Haringey is in north London, England.
Sentence 1: The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.
  Fact 1.0: The Mall Wood Green is a large shopping centre.
  Fact 1.1: The Mall Wood Green is a residential complex.
  Fact 1.2: The Mall Wood Green is in Wood Green, north London.

Retrieved items:
<item 1>
London Skolars
London Skolars R.L.F.C. are a professional rugby league club based at the New River Stadium, Wood Green, Haringey in north London.

 They were founded in 1995 and have been professional since 2003, operating in the Kingstone Press League 1 in the 2015 season.

 They also run an A-team (formerly known as "Haringey Hornets") that play in the South Premier Division of the Rugby League Conference.
</item 1>

<item 2>
Haringey Heartlands
Haringey Heartlands is a currently ongoing urban regeneration project in the London Borough of Haringey, in north London, England.

 The site is located west of Shopping City in Wood Green.

 The aim of the development is "to create a vibrant and attractive new urban quarter which acts as the civic and cultural Heart of Haringey, integrating with
and benefiting wider communities."
</item 2>

<item 3>
The Mall Wood Green
The Mall Wood Green is a large shopping centre and residential complex in Wood Green, north London.

 It is generally still referred to by its former name of Wood Green Shopping City; the signage on the building still uses that name.
</item 3>

<item 4>
Wood Green tube station
Wood Green is a London Underground station on the Piccadilly line.

 The station is between Turnpike Lane and Bounds Green stations and is in Travelcard Zone 3.

 It is located at junction of High Road, Wood Green and Lordship Lane.

 It serves Wood Green Shopping City and the nearby Haringey Council administrative complex as well as a densely populated residential area.

 It is also the closest tube station to the Alexandra Palace Venue.
</item 4>

<item 5>
Wood Green War Memorial
The Wood Green War Memorial is located on the High Road, Wood Green, in the London Borough of Haringey.

 It was erected in 1920 and paid for by public subscription of the people of Wood Green to remember the men of the area who died during the First World War.

 It was subsequently adapted to include the dead of the Second World War.

 It is grade II listed with Historic England.
</item 5>

<item 6>
White Hart Lane Community Sports Centre
White Hart Lane Community Sports Centre, also known as the New River Stadium is a rugby league and athletics stadium in Wood Green, Haringey, north London,
England that is home to London Skolars rugby league club, Enfield and Haringey Athletic Club, Wood Green Weightlifting Club, Hashtag United F.C. (YouTube based
football club), Haringey Rhinos rugby union club, Haringey Cycling Club, and the Next Level Football League.
</item 6>

<item 7>
Haringey London Borough Council
Haringey London Borough Council is the local authority for the London Borough of Haringey in Greater London, England.

 It is a London borough council, one of 32 in the United Kingdom capital of London.

 Haringey is divided into 19 wards, each electing three councillors.

 Haringey London Borough Council comprises 48 Labour Party councillors, and 9 Liberal Democrats.

 The council was created by the London Government Act 1963 and replaced three local authorities: Hornsey Borough Council, Tottenham Borough Council and Wood
Green Borough Council.

 The next election to the authority will be in 2018.
</item 7>

<item 8>
Heartlands High School
Heartlands High School is a state secondary school located in Wood Green in the London Borough of Haringey.

 It welcomed its first students in September 2010.

 The headteacher is Mr. Simon Garrill.
</item 8>

<item 9>
Wood Green Town F.C.
Wood Green Town Football Club was a football club based in Tottenham in the London Borough of Haringey, England.

 Formed in 1911 as a breakaway from Tufnell Park, they merged with Edmonton (a descendent of Tufnell Park) in 1973 to form Edmonton & Haringey.

 During their history the club were known as Tufnell Spartans, Wood Green, Wood Green Town and Haringey Borough.
</item 9>

<item 10>
Alexandra Palace railway station
Alexandra Palace railway station (originally named Wood Green and later Wood Green (Alexandra Park)) is in the London Borough of Haringey in north London, and
is in Zone 3.

 The station, and all trains serving it, are operated by Great Northern.

 Trains mostly go to and from Moorgate on weekdays and weekends.
</item 10>
=== END ITEM id=258cb6a35d0a63fe ===

=== ITEM id=966d61e788d84922 ===
Gold sentences and facts:
Sentence 0: The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.
  Fact 0.0: The presidency of Ronald Reagan began on January 20, 1981
  Fact 0.1: Ronald Reagan was inaugurated as President of the United States on January 20, 1981
  Fact 0.2: The presidency of Ronald Reagan ended on January 20, 1989
Sentence 1: George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice President of the United States from 1981 to 1989.
  Fact 1.0: George Herbert Walker Bush was born June 12, 1924
  Fact 1.1: George Herbert Walker Bush is an American politician
  Fact 1.2: George Herbert Walker Bush was the 41st President of the United States from 1989 to 1993
  Fact 1.3: George Herbert Walker Bush was the 43rd Vice President of the United States from 1981 to 1989

Retrieved items:
<item 1>
First inauguration of Ronald Reagan
The first inauguration of Ronald Reagan as the 40th President of the United States was held on Tuesday, January 20, 1981, on the West Front of the United States
Capitol in Washington, D.C..

 This was the first inauguration to be held on the building's west side.

 The inauguration marked the commencement of the first four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.

 Chief Justice Warren E. Burger administered the presidential oath of office to Reagan, who placed his hand upon a family Bible given to him by his mother, open
to .

 Associate Justice Potter Stewart administered the vice presidential oath to Bush.
</item 1>

<item 2>
Second inauguration of Ronald Reagan
The second inauguration of Ronald Reagan as President of the United States was held privately on Sunday, January 20, 1985 in the Grand Foyer of the White House,
and publicly the following day, January 21, 1985 at the United States Capitol in Washington, D.C. The inauguration marked the commencement of the second
four-year term of Ronald Reagan as President and of George H. W. Bush as Vice President.
</item 2>

<item 3>
Ronald Reagan judicial appointment controversies
During President Ronald Reagan's presidency, he nominated at least twelve people for various federal appellate judgeship who were not confirmed.

 In some cases, the nominations were not processed by the Democratic-controlled Senate Judiciary Committee before Reagan's presidency ended, while in other
cases, nominees were rejected by the Senate Judiciary Committee or even blocked by unfriendly members of the Republican Party.

 Three of the nominees were renominated by Reagan's successor, President George H. W. Bush.

 Two of the nominees, Ferdinand Francis Fernandez and Guy G. Hurlbutt, were nominated after July 1, 1988, the traditional start date of the unofficial Thurmond
Rule during a presidential election year.

 Eight of the twelve seats eventually were filled by appointees of President George H. W. Bush.
</item 3>

<item 4>
Otto Reich
Otto Juan Reich (born October 16, 1945), is an American former senior official in the administrations of Presidents Ronald Reagan, George H. W. Bush, and George
W. Bush.

 Among other positions held, he has been the President's Special Envoy for the Western Hemisphere; Assistant Secretary of State for Western Hemisphere Affairs;
United States Ambassador to Venezuela; and Assistant Administrator of the US Agency for International Development, a recess appointment.

 In 2003, Bush appointed him US Special Envoy to the Western Hemisphere.

 Since leaving the White House in 2004, he has headed his own international consulting firm, Otto Reich Associates, LLC, based in Washington, D.C..
</item 4>

<item 5>
Elliott Abrams
Elliott Abrams (born January 24, 1948) is a former American diplomat, lawyer and political scientist who served in foreign policy positions for Presidents
Ronald Reagan and George W. Bush.

 Abrams was convicted of withholding information from Congress about the Iran–Contra affair while serving under Reagan, but was pardoned by President George H.
W. Bush.
</item 5>

<item 6>
Presidency of Ronald Reagan
The presidency of Ronald Reagan began on January 20, 1981, when Ronald Reagan was inaugurated as President of the United States, and ended on January 20, 1989.

 Reagan, a Republican, took office as the 40th United States president following a landslide win over Democratic incumbent President Jimmy Carter in the 1980
presidential election.

 Reagan was succeeded by his vice president, George H. W. Bush, who won the 1988 presidential election with Reagan's support.
</item 6>

<item 7>
George H. W. Bush
George Herbert Walker Bush (born June 12, 1924) is an American politician who was the 41st President of the United States from 1989 to 1993 and the 43rd Vice
President of the United States from 1981 to 1989.

 A member of the Republican Party, he was previously a congressman, ambassador, and Director of Central Intelligence.

 He is the oldest living former President and Vice President.

 Previously known as simply "George Bush", since 2001, Bush has often been referred to as "George H. W. Bush", "Bush 41", "Bush the Elder", or "George Bush
Senior" to distinguish him from his eldest son, George W. Bush, the 43rd President of the United States.
</item 7>

<item 8>
Reagan Democrat
A Reagan Democrat is a traditionally Democratic voter in the United States, referring especially to white working-class Rust Belt residents, who defected from
their party to support Republican President Ronald Reagan in either or both of the 1980 and 1984 elections as well as Republican Presidents George H. W. Bush in
the 1988 election and George W. Bush in either or both of the 2000 and 2004 elections.

 Part of this group also defected to Donald Trump, the Republican candidate in 2016, who won in Pennsylvania, Ohio, Michigan, Wisconsin, and Iowa, all states
that voted for Reagan but went Democratic in 2008 and 2012.
</item 8>

<item 9>
Gerald Ford
Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the
United States from August 1974 to January 1977, following the resignation of Richard Nixon.

 Prior to this he served eight months as the 40th Vice President of the United States, following the resignation of Spiro Agnew.

 He was the first person appointed to the vice presidency under the terms of the 25th Amendment, and consequently the only person to have served as both Vice
President and President of the United States without being elected to executive office.

 Before his appointment to the vice presidency, Ford served 25 years as U.S. Representative from Michigan's 5th congressional district, the final nine of them
as the House Minority Leader.
</item 9>

<item 10>
William Flynn Martin
William Flynn Martin (born October 4, 1950) is an American energy economist, educator and international diplomat.

 Martin served as Special Assistant to President Reagan for National Security Affairs, Executive Secretary of the National Security Council in the West Wing of
the White House and Deputy Secretary of the Department of Energy during the Ronald Reagan administration.

 He was President of the Council of the University for Peace, appointed to the Council by Secretary General of the United Nations Kofi Annan and served as the
Executive Director of the Republican Platform Committee during the re-election bid of George H.W. Bush.

 He has held senior appointments and advisory positions under several Presidents including: Ronald Reagan, George H.W. Bush and George W. Bush.
</item 10>
=== END ITEM id=966d61e788d84922 ===

=== ITEM id=8462100b2e1b3c1b ===
Gold sentences and facts:
Sentence 0: The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of the Stones River Campaign in the Western Theater of the American Civil War.
  Fact 0.0: The Battle of Stones River is also called the Second Battle of Murfreesboro.
  Fact 0.1: The Battle of Stones River was fought from December 31, 1862, to January 2, 1863.
  Fact 0.2: The Battle of Stones River was fought in Middle Tennessee.
  Fact 0.3: The Battle of Stones River was the culmination of the Stones River Campaign.
  Fact 0.4: The Stones River Campaign took place in the Western Theater of the American Civil War.
Sentence 1: The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.
  Fact 1.0: The Battle of the Ch'ongch'on River is also known as the Battle of the Ch'ongch'on.
  Fact 1.1: The Battle of the Ch'ongch'on River is also known as the Second Phase Campaign Western Sector.
  Fact 1.2: The Battle of the Ch'ongch'on River was a decisive battle in the Korean War.
  Fact 1.3: The Battle of the Ch'ongch'on River took place from November 25 to December 2, 1950.
  Fact 1.4: The Battle of the Ch'ongch'on River took place along the Ch'ongch'on River Valley.
  Fact 1.5: The Ch'ongch'on River Valley is in the northwestern part of North Korea.

Retrieved items:
<item 1>
Seven Days Union order of battle
The following Union Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization-return of casualties during the battle and the reports.

 The Confederate order of battle is listed separately.
</item 1>

<item 2>
Battle of Stones River
The Battle of Stones River or Second Battle of Murfreesboro, was fought from December 31, 1862, to January 2, 1863, in Middle Tennessee, as the culmination of
the Stones River Campaign in the Western Theater of the American Civil War.

 Of the major battles of the Civil War, Stones River had the highest percentage of casualties on both sides.

 Although the battle itself was inconclusive, the Union Army's repulse of two Confederate attacks and the subsequent Confederate withdrawal were a much-needed
boost to Union morale after the defeat at the Battle of Fredericksburg, and it dashed Confederate aspirations for control of Middle Tennessee.
</item 2>

<item 3>
Korean War
The Korean War (in South Korean Hangul: 한국전쟁 ; Hanja: 韓國戰爭 ; RR: "Hanguk Jeonjaeng " , "Korean War"; in North Korean Chosŏn'gŭl: 조국해방전쟁 ; Hancha: 祖國解放戰爭 ; MR:
"Choguk haebang chǒnjaeng " , "Fatherland Liberation War"; 25 June 1950 – 27 July 1953) was a war between North Korea (with the support of China and the Soviet
Union) and South Korea (with the principal support of the United States).

 The war began on 25 June 1950 when North Korea invaded South Korea.

 The United Nations, with the United States as the principal force, came to the aid of South Korea.

 China came to the aid of North Korea, and the Soviet Union also gave some assistance to the North.
</item 3>

<item 4>
Battle of Gaines's Mill
The Battle of Gaines's Mill, sometimes known as the First Battle of Cold Harbor or the Battle of Chickahominy River, took place on June 27, 1862, in Hanover
County, Virginia, as the third of the Seven Days Battles (Peninsula Campaign) of the American Civil War.

 Following the inconclusive Battle of Beaver Dam Creek (Mechanicsville) the previous day, Confederate General Robert E. Lee renewed his attacks against the
right flank of the Union Army, relatively isolated on the northern side of the Chickahominy River.

 There, Brig. Gen. Fitz John Porter's V Corps had established a strong defensive line behind Boatswain's Swamp.

 Lee's force was destined to launch the largest Confederate attack of the war, about 57,000 men in six divisions.

 Porter's reinforced V Corps held fast for the afternoon as the Confederates attacked in a disjointed manner, first with the division of Maj. Gen. A.P. Hill,
then Maj. Gen. Richard S. Ewell, suffering heavy casualties.

 The arrival of Maj. Gen. Stonewall Jackson's command was delayed, preventing the full concentration of Confederate force before Porter received some
reinforcements from the VI Corps.
</item 4>

<item 5>
Battle of Cross Keys
The Battle of Cross Keys was fought on June 8, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Together, the battles of Cross Keys and Port Republic the following day were the decisive victories in Jackson's Valley Campaign, forcing the Union armies to
retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 5>

<item 6>
King Kang of Chu
King Kang of Chu (, died 545 BC) was from 559 to 545 BC the king of Chu, a major power during the Spring and Autumn period of ancient China.

 Born Xiong Zhao (), he succeeded his father King Gong of Chu.

 His reign was marked by constant wars with Jin, Chu's traditional enemy, and Wu, its new enemy.

 He ruled for 15 years and was succeeded by his son, Jia'ao, who would four years later be murdered by King Kang's younger brother King Ling of Chu.
</item 6>

<item 7>
Seven Days Confederate order of battle
The following Confederate States Army units and commanders fought in the Seven Days Battles (from June 25 to July 1, 1862) of the American Civil War.

 Order of battle compiled from the army organization during the battle, the casualty returns and the reports.

 The Union order of battle is listed separately.
</item 7>

<item 8>
Battle of Port Republic
The Battle of Port Republic was fought on June 9, 1862, in Rockingham County, Virginia, as part of Confederate Army Maj. Gen. Thomas J. "Stonewall" Jackson's
campaign through the Shenandoah Valley during the American Civil War.

 Port Republic was a fierce contest between two equally determined foes and was the most costly battle fought by Jackson's Army of the Valley during its
campaign.

 Together, the battles of Cross Keys (fought the previous day) and Port Republic were the decisive victories in Jackson's Valley Campaign, forcing the Union
armies to retreat and leaving Jackson free to reinforce Gen. Robert E. Lee for the Seven Days Battles outside Richmond, Virginia.
</item 8>

<item 9>
Seven Days Battles
The Seven Days Battles were a series of six major battles over the seven days from June 25 to July 1, 1862, near Richmond, Virginia, during the American Civil
War.

 Confederate General Robert E. Lee drove the invading Union Army of the Potomac, commanded by Maj. Gen. George B. McClellan, away from Richmond and into a
retreat down the Virginia Peninsula.

 The series of battles is sometimes known erroneously as the Seven Days Campaign, but it was actually the culmination of the Peninsula Campaign, not a separate
campaign in its own right.
</item 9>

<item 10>
Battle of the Ch'ongch'on River
The Battle of the Ch'ongch'on River, also known as the Battle of the Ch'ongch'on () or the Second Phase Campaign Western Sector (), was a decisive battle in the
Korean War, and it took place from November 25 to December 2, 1950, along the Ch'ongch'on River Valley in the northwestern part of North Korea.

 In response to the successful Chinese First Phase Campaign against the United Nations forces, General Douglas MacArthur launched the Home-by-Christmas
Offensive to evict the Chinese forces from Korea and to end the war.

 Anticipating this reaction, the Chinese People's Volunteer Army Commander Peng Dehuai planned a counteroffensive, dubbed the "Second Phase Campaign", against
the advancing UN forces.
</item 10>
=== END ITEM id=8462100b2e1b3c1b ===

=== ITEM id=b0f9f864359cd64b ===
Gold sentences and facts:
Sentence 0: The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
  Fact 0.0: The Apple Dumpling Gang is a 1975 American comedy-western film.
  Fact 0.1: The Apple Dumpling Gang was produced by Walt Disney Productions.
  Fact 0.2: The Apple Dumpling Gang is about a slick gambler named Russell Donovan.
  Fact 0.3: Russell Donovan is played by Bill Bixby.
  Fact 0.4: Russell Donovan is duped into taking care of a group of orphans.
  Fact 0.5: The group of orphans eventually strikes gold during the California Gold Rush.
Sentence 1: Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.
  Fact 1.0: Heavyweights is a 1995 American comedy-drama film.
  Fact 1.1: Heavyweights was directed by Steven Brill.
  Fact 1.2: Heavyweights was written by Steven Brill with Judd Apatow.

Retrieved items:
<item 1>
The entries cover documentaries on Boston hardcore (All Ages, 2012), punk (Punk’s Not Dead, 2007; American Hardcore, 2006), bipolar disorder, artistic aging,
slavery, archaeology, bookstores, and Charles Bukowski; plus Barbara Hammer’s autobiographical Tender Fictions. Fiction films include Bakshi’s acclaimed X-rated
Heavy Traffic, Heavyweights, several American Pie and Bring It On sequels, Schrader’s Hardcore, Blue Collar, and The
</item 1>

<item 2>
The Sasquatch Gang
The Sasquatch Gang (also known as The Sasquatch Dumpling Gang) is a 2006 comedy film written and directed by Tim Skousen, the first assistant director on
"Napoleon Dynamite".
</item 2>

<item 3>
Freebie and the Bean
Freebie and the Bean is a 1974 American action-comedy film about two off-beat police detectives who wreak havoc in San Francisco attempting to bring down a
local organized crime boss.

 The picture, a precursor to the buddy cop film genre popularized a decade later, stars James Caan, Alan Arkin, Loretta Swit and Valerie Harper.

 Harper was nominated for the Golden Globe for New Star of the Year for playing the Hispanic wife of Alan Arkin.

 The film was directed by Richard Rush.

 An article in Rolling Stone magazine alleged that Stanley Kubrick called "Freebie and the Bean" the best film of 1974.

 Arkin and Caan would not appear in another movie together until the 2008 film adaptation of "Get Smart".
</item 3>

<item 4>
Heavyweights
Heavyweights is a 1995 American comedy-drama film directed by Steven Brill and written by Brill with Judd Apatow.

 The film centers around a fat camp for kids that is taken over by a fitness guru named Tony Perkis (Ben Stiller).
</item 4>

<item 5>
The Apple Dumpling Gang (film)
The Apple Dumpling Gang is a 1975 American comedy-western film produced by Walt Disney Productions about a slick gambler named Russell Donovan (Bill Bixby) who
is duped into taking care of a group of orphans who eventually strike gold during the California Gold Rush.
</item 5>

<item 6>
The entries span adult animation, comedy, drama, westerns, science fiction, family films, and television. Ralph Bakshi’s *Fritz the Cat* (1972), the first U.S.
animated feature rated X, and *Fire and Ice* (1983) contrast with his live-action *Cool and the Crazy*. Disney titles include *The Apple Dumpling Gang* films,
*Darby O’Gill*, and *Home on the Range*. Other franchises include *The Naked
</item 6>

<item 7>
All Ages: The Boston Hardcore Film
All Ages: The Boston Hardcore Film, a Gallery East Production in association with Stone Films/NYC, is a documentary directed by Drew Stone, which had its world
premier at the Independent Film Festival of Boston on April 27, 2012.

 The film features interviews, archival footage and the music of Boston’s early hardcore bands including Deep Wound, DYS, Gang Green, Impact Unit, Jerry's Kids,
Negative FX, SS Decontrol, The Freeze and The F.U.'s.

 Also featured in the film are interviews with renowned author Michael Patrick MacDonald ("Easter Rising", "All Souls"), Actress Christine Elise McCarthy,
Advertising Executive Jonathan Anastas, Thrasher Magazine Editor Jake Phelps, "American Hardcore" Director Paul Rachman, and Newbury Comics owner Michael
Dreese.

 "All Ages" has been released on DVD with numerous extras and director's commentary.
</item 7>

<item 8>
American Hardcore (film)
American Hardcore: The History of American Punk Rock 1980-1986 is a documentary directed by Paul Rachman and written by Steven Blush.

 It is based on the book "" also written by Blush.

 It was released on September 22, 2006 on a limited basis.

 The film features some early pioneers of the hardcore punk music scene including Bad Brains, Black Flag, D.O.A., Minor Threat, Minutemen, SSD, and others.

 It was released on DVD by Sony Pictures Home Entertainment on February 20, 2007.
</item 8>

<item 9>
The Apple Dumpling Gang
The Apple Dumpling Gang is a 1971 novel by Jack Bickham, about a group of orphaned children during the California gold rush.

 They encounter a gambler who reluctantly helps them, as well as a pair of hapless robbers who are after the gold the children have found.
</item 9>

<item 10>
The Apple Dumpling Gang Rides Again
The Apple Dumpling Gang Rides Again is a 1979 American comedy-western film produced by Walt Disney Productions and a sequel to "The Apple Dumpling Gang" (1975),
starring the comedy duo of Tim Conway and Don Knotts reprising their respective roles as Amos and Theodore.

 The film also stars Tim Matheson, Harry Morgan, and Kenneth Mars.

 Ruth Buzzi appears in a small cameo as a wild farsighted woman.

 Robert Totten, who directed installments of "Gunsmoke", also had a small part in the film.
</item 10>
=== END ITEM id=b0f9f864359cd64b ===

=== ITEM id=4d092192d2e9d1ba ===
Gold sentences and facts:
Sentence 0: The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
  Fact 0.0: The Silent Historian has the original title 'Het zwijgen van Loe de Jong [The Silence of Loe de Jong]'.
  Fact 0.1: The Silent Historian is a 2011 Dutch documentary film.
  Fact 0.2: The Silent Historian is a documentary film by Simonka de Jong.
  Fact 0.3: The Silent Historian was shown as part of the January 2012 New York Jewish Film Festival.
  Fact 0.4: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Jewish Museum.
  Fact 0.5: The January 2012 New York Jewish Film Festival screening of The Silent Historian was presented in partnership with the Film Society of Lincoln Center.
Sentence 1: The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
  Fact 1.0: The Betrayal — Nerakhoon is a 2008 documentary film.
  Fact 1.1: The Betrayal — Nerakhoon was directed by Ellen Kuras.
  Fact 1.2: The Betrayal — Nerakhoon was directed by Thavisouk Phrasavath.

Retrieved items:
<item 1>
Illusions &amp; Mirrors
Illusions & Mirrors is a 2013 short film directed by Iranian-born American artist and filmmaker Shirin Neshat.

 The film, shot in black-and-white, stars Tarek Aylouch, Michael Markiewicz, and the Israeli-born American actress Natalie Portman.

 The film was commissioned by Dior, for which Portman is a spokesmodel, for the Miss Dior Expo at the Grand Palais, Paris.

 Shirin Neshat said that the film is a tribute to the black-and-white silent films made by such surrealist filmmakers as Man Ray, Jean Cocteau, Luis Buñuel, and
Maya Deren.
</item 1>

<item 2>
The Counterfeiters (2007 film)
The Counterfeiters (German: "Die Fälscher" ) is a 2007 Austrian-German drama film written and directed by Stefan Ruzowitzky.

 It fictionalizes Operation Bernhard, a secret plan by Nazi Germany during World War II to destabilize the United Kingdom by flooding its economy with forged
Bank of England pound notes.

 The film centres on a Jewish counterfeiter, Salomon 'Sally' Sorowitsch, who is coerced into assisting the operation at the Sachsenhausen concentration camp.
</item 2>

<item 3>
No Names on the Doors
No Names on the Doors (Hebrew: אין שמות על הדלתות‎ ) is a 1997 Israeli drama directed by Nadav Levitan.

 The film was screened at the Berlin International Film Festival, the Montpellier Film Festival, the Shanghai International Film Festival, and the Melbourne
International Film Festival.

 It received two nominations for an Ophir Award in the categories of Best Actor and Best Screenplay.
</item 3>

<item 4>
Thérèse Raquin (1928 film)
Thérèse Raquin is a 1928 drama film directed by Jacques Feyder.

 It is the third silent film adaptation of the novel of the same name by Émile Zola.

 The film stars Gina Manès as Thérèse Raquin, Wolfgang Zilzer as Monsieur Raquin, and Jeanne Marie Laurent as Madame Raquin.

 The décors of the Paris suburbs for the film were built by André Andrejew.

 The film was produced by Deutsche Film Union in Germany, with German and French actors, in a French-German co-production, to be later released at the same time
in France as "Thérèse Raquin" and Germany as Du sollst nicht ehebrechen!

.

 As no words were spoken, both versions differed only with the language of intertitles.

 The British title at the time of the film's original release was Thou Shalt Not.

 This is last silent film imports distributed by Warner Bros.' newly acquired First National subsidiary.

 No dialong with music score and sound effects.
</item 4>

<item 5>
The Deal (2015 film)
The Deal (; lit.

 "Murder Request") is a 2015 South Korean crime thriller film directed by Son Yong-ho.
</item 5>

<item 6>
The Price of Betrayal
The Price of Betrayal (Swedish: "Judaspengar" ) is a 1915 Swedish silent drama film directed by Victor Sjöström.
</item 6>

<item 7>
Betrayal (1929 film)
Betrayal was a 1929 drama film produced for Famous Players-Lasky and released by Paramount Pictures.

 The film is the last silent film directed by Lewis Milestone, the last silent performance by Gary Cooper, the last silent performance by Germany's Emil
Jannings, and the only onscreen pairing of Cooper and Jannings.

 It is considered a lost film.
</item 7>

<item 8>
The Betrayal – Nerakhoon
The Betrayal — Nerakhoon is a 2008 documentary film directed by Ellen Kuras and Thavisouk Phrasavath.
</item 8>

<item 9>
The Silent Historian
The Silent Historian (original title: "Het zwijgen van Loe de Jong [The Silence of Loe de Jong]") is a 2011 Dutch documentary film by Simonka de Jong, shown as
part of the January 2012 New York Jewish Film Festival, presented in partnership with Jewish Museum and Film Society of Lincoln Center.
</item 9>

<item 10>
The Decembrists (film)
The Decembrists (Russian: Декабристы , "Dekabristy " ) is a 1927 Soviet silent historical drama film directed by Aleksandr Ivanovsky.
</item 10>
=== END ITEM id=4d092192d2e9d1ba ===

=== ITEM id=59bb851255d87feb ===
Gold sentences and facts:
Sentence 0:  He has been involved in numerous television projects as director, writer, producer and animator.
  Fact 0.0: The person described has been involved in numerous television projects.
  Fact 0.1: The person described has worked on television projects as a director.
  Fact 0.2: The person described has worked on television projects as a writer.
  Fact 0.3: The person described has worked on television projects as a producer.
  Fact 0.4: The person described has worked on television projects as an animator.
Sentence 1: Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.
  Fact 1.0: Bela Gaal was born on 2 January 1893.
  Fact 1.1: Bela Gaal died on 18 February 1945.
  Fact 1.2: Bela Gaal was a Hungarian film director.

Retrieved items:
<item 1>
Fire and Ice (1983 film)
Fire and Ice is a 1983 American adult animated epic high fantasy adventure film directed by Ralph Bakshi.

 The film, a collaboration between Bakshi and Frank Frazetta, was distributed by 20th Century Fox, which also distributed Bakshi's 1977 release, "Wizards".

 The animated feature, based on characters Bakshi and Frazetta co-created, was made using the process of rotoscoping, in which scenes were shot in live action
and then traced onto animation cels.
</item 1>

<item 2>
Cool and the Crazy
Cool and the Crazy is a 1994 American drama film written and directed by Ralph Bakshi and starring Jared Leto and Alicia Silverstone.

 The story revolves around an unhappily married couple in the late 1950s who both lead separate affairs.

 The film was Bakshi's first feature-length live-action film, being primarily known as a director of animated films which heavily utilize live-action sequences,
such as "Fritz the Cat", "Heavy Traffic", "Wizards", "American Pop" and "The Lord of the Rings".
</item 2>

<item 3>
Béla Gaál
Béla Gaál (2 January 1893 – 18 February 1945) was a Hungarian film director.

 His 1930 film "Csak egy kislány van a világon" was the first sound film to be made in Hungary.
</item 3>

<item 4>
Epic Pooh
"Epic Pooh" is a 1978 article by the British science fiction writer Michael Moorcock, which reviews the field of epic fantasy, with a particular focus on epic
fantasy written for children.

 In it Moorcock critiques J. R. R. Tolkien's "The Lord of the Rings" for its politically conservative assumptions and for being escapist literature.
</item 4>

<item 5>
Tales of Budapest
Tales Of Budapest (Hungarian: "Pesti mese" ) is a 1937 Hungarian comedy film directed by Béla Gaál.
</item 5>

<item 6>
Heavy Traffic
Heavy Traffic is a 1973 American adult animated comedy-drama film written and directed by Ralph Bakshi.

 The film, which begins, ends, and occasionally combines with live-action, explores the often surreal fantasies of a young New York cartoonist named Michael
Corleone, using pinball imagery as a metaphor for inner-city life.

 "Heavy Traffic" was Bakshi and producer Steve Krantz's follow-up to the film "Fritz the Cat".

 Though producer Krantz made varied attempts to produce an R-rated film, "Heavy Traffic" was given an X rating by the MPAA.

 The film received positive reviews and is widely considered to be Bakshi's biggest critical success.
</item 6>

<item 7>
Ralph Bakshi
Ralph Bakshi (born October 29, 1938) is an American director of animated and live-action films.

 In the 1970s, he established an alternative to mainstream animation through independent and adult-oriented productions.

 Between 1972 and 2015, he directed ten theatrically released feature films, six of which he wrote.

 He has been involved in numerous television projects as director, writer, producer and animator.
</item 7>

<item 8>
Peter Jackson
Sir Peter Robert Jackson {'1': ", '2': ", '3': ", '4': "} (born 31 October 1961) is a New Zealand film director, screenwriter and film producer.

 He is best known as the director, writer, and producer of "The Lord of the Rings" trilogy (2001–03) and "The Hobbit" trilogy (2012–14), both of which are
adapted from the novels of the same name by J. R. R. Tolkien.

 Other notable films include the critically lauded drama "Heavenly Creatures" (1994), the mockumentary "Forgotten Silver" (1995), the horror comedy "The
Frighteners" (1996), the epic monster remake film "King Kong" (2005), and the supernatural drama film "The Lovely Bones" (2009).

 He also produced "District 9" (2009), "The Adventures of Tintin: The Secret of the Unicorn" (2011), and the documentary "West of Memphis" (2012).
</item 8>

<item 9>
Fritz the Cat (film)
Fritz the Cat is a 1972 American adult animated comedy film written and directed by Ralph Bakshi.

 It was Bakshi's feature film debut and is loosely based on the Fritz the Cat comic strips by Robert Crumb.

 It was the first animated feature film to receive an X rating in the United States.
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
=== END ITEM id=59bb851255d87feb ===

=== ITEM id=b66a37e0a94a4b17 ===
Gold sentences and facts:
Sentence 0: Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.
  Fact 0.0: Pulandian District is one of the seven districts under the administration of Dalian.
  Fact 0.1: Pulandian District is located in the south of Liaoning province.
  Fact 0.2: Liaoning province is in the People's Republic of China.
Sentence 1: Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.
  Fact 1.0: Kaiyuan is a county-level city.
  Fact 1.1: Kaiyuan is in the northeast of Liaoning.
  Fact 1.2: Liaoning is in the People's Republic of China.
  Fact 1.3: Kaiyuan borders Jilin for a small section to the north.

Retrieved items:
<item 1>
Wenling
Wenling (Wenling dialect: Uen-lin Zy ] ; ) is a coastal county-level city in the municipal region of Taizhou, in southeastern Zhejiang province, China.

 It borders Luqiao and Huangyan to the north, Yuhuan to the south, Yueqing to the west, looks out to the East China Sea to the east.

 Wenling locates on 28°22'N, 121°21'E, approximately 300 km south of Shanghai.
</item 1>

<item 2>
Pulandian District
Pulandian District () is one of the seven districts under the administration of Dalian, located in the south of Liaoning province, People's Republic of China.

 Its area is 2769.90 km² and its permanent population as of 2010 is 741,230.

 The district borders the prefecture-level city of Yingkou to the north.
</item 2>

<item 3>
Shuangta, Pulandian
Shuangta () is a town under the administration of Pulandian City in southern Liaoning province, China, located 58 km northeast of downtown Pulandian and about
120 km northeast of Dalian.

 , it has 9 villages under its administration.
</item 3>

<item 4>
Xingcheng
Xingcheng (), former name Ningyuan (宁远), is a county-level city of southwest Liaoning province, China, with a population of approximately 140,000 urban
inhabitants, and is located on the Liaodong Bay, i.e. the northern coast of the Bohai Sea.

 Currently under the administration of Huludao City, the area is steeped in history, and contains one of the best preserved Ming Dynasty towns in China, as well
as functioning as a laidback summer resort.
</item 4>

<item 5>
Fuzhoucheng
Fuzhou, also Fuzhoucheng () is a town under the administration of Wafangdian City, in southern Liaoning province, China.

 It lies at the intersection of China National Highway 202 and Liaoning Provincial Highway 313, 57 km north by road from Pulandian.

 Dahe Reservoir is located several kilometres to the east.

 There is a 74,132 acres (30,000 hectares) bird reserve nearby with species such as Swan goose and Hooded crane about 15 km southwest of the town.

 , it has 2 residential communities (社区) and 14 villages under its administration.
</item 5>

<item 6>
Kaiyuan West Railway Station
Kaiyuan West Railway Station is a railway station of Hada Passenger Railway and located in Liaoning, China
</item 6>

<item 7>
Longkou
Longkou (), formerly Huang County (), is a port city in northeastern Shandong province, China, facing the Bohai Sea to the north and the Laizhou Bay to the
west.

 Longkou, a county-level city, is administered by the prefecture-level city of Yantai.

 The total population of Longkou is 620,000.
</item 7>

<item 8>
Qinghe District, Tieling
Qinghe District () is a district of Tieling, Liaoning province, People's Republic of China.

 However, it is located closer to Kaiyuan City than it is to downtown Tieling, which lies 39 km to the southwest.
</item 8>

<item 9>
Kaiyuan, Liaoning
Kaiyuan () is a county-level city in the northeast of Liaoning, People's Republic of China, bordering Jilin for a small section to the north.

 It is under the administration of Tieling City, the centre of which lies 33 km to the southwest.
</item 9>

<item 10>
Taiyang Subdistrict
Taiyang Subdistrict () is a subdistrict of Wafangdian, Dalian District, Liaoning, China.

 It is based around the small town of Taiyang, Taiyangsheng or Taiyangshengxiang, which lies 19 kilometres east by road of Fuzhou, 64 kilometres north by road
from Pulandian.

 To the east is the DFM reservoir.
</item 10>
=== END ITEM id=b66a37e0a94a4b17 ===

=== ITEM id=a1fce7d83a11f849 ===
Gold sentences and facts:
Sentence 0: Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and Vietnam.
  Fact 0.0: Aspidistra is a genus of flowering plants.
  Fact 0.1: Aspidistra is in the family Asparagaceae.
  Fact 0.2: Aspidistra is in the subfamily Nolinoideae.
  Fact 0.3: Aspidistra is native to eastern Asia.
  Fact 0.4: Aspidistra is native to southeastern Asia.
  Fact 0.5: Aspidistra is particularly native to China.
  Fact 0.6: Aspidistra is particularly native to Vietnam.
Sentence 1: Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
  Fact 1.0: Cyrtanthus is a genus of perennial plants.
  Fact 1.1: Cyrtanthus is a genus of herbaceous plants.
  Fact 1.2: Cyrtanthus is a genus of bulbous plants.
  Fact 1.3: Cyrtanthus is in the family Amaryllidaceae.
  Fact 1.4: Cyrtanthus is in the subfamily Amaryllidoideae.

Retrieved items:
<item 1>
Caryophyllales
Caryophyllales ( ) is an order of flowering plants that includes the cacti, carnations, amaranths, ice plants, beets, and many carnivorous plants.

 Many members are succulent, having fleshy stems or leaves.
</item 1>

<item 2>
Clinantheae
Clinantheae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was described in 2000 by Alan Meerow "et al." as a result of a molecular phylogenetic study of the American Amaryllidoideae.

 This demonstrated that the Stenomesseae tribe, including the type genus "Stenomesson" was polyphyletic.

 Part of the tribe segregated with the Eucharideae and were submerged into it, while the other part formed a unique subclade.

 Since the type species of "Stenomesson" was not part of the second sublclade it was necessary to form a new name for the remaining species together with the
other genera that remained.

 This was "Clinanthus", the oldest name for these species, and consequently the tribe Clinantheae.
</item 2>

<item 3>
Graph query row:
genus: Aspidistra
relationship: PART_OF
taxonomic_connection: Nolinoideae
kind: subfamily
</item 3>

<item 4>
Hymenocallideae
Hymenocallideae is a tribe (in the family Amaryllidaceae, subfamily Amaryllidoideae), where it forms part of the Andean clade, one of two American clades.

 The tribe was originally recognised by both Meerow (1995) and the Muller-Doblies' (1996).

 Its phylogenetic position within the Amaryllidoideae was established by Meerow "et al." in 2000, while in-depth infratribal relationships were established in
2002.
</item 4>

<item 5>
Cyrtanthus
Cyrtanthus is a genus of perennial, herbaceous and bulbous plants in the family Amaryllidaceae, subfamily Amaryllidoideae.
</item 5>

<item 6>
Amarylloidinae
Amarylloidinae is a now obsolete informal name for an "infrafamily" within the Amaryllidaceae (Amaryllis) family, erected by Hamilton Traub.

 This grouping was designed to fill a perceived gap between the formal rank of subfamily and tribes.

 In his treatment of this family, he divided it first into four subfamiles.

 Within subfamily Amarylloideae he then divided his sixteen tribes into two infrafamilies, Amarylloidinae (12 tribes) and Pancratioidinae (4 tribes), both of
which were subsequently demonstrated to be polyphyletic, and hence were abandoned by Dahlgren, who used no rank between family and tribe.

 On the other hand, he also used a much more restricted Amaryllidaceae corresponding to Traub's subfamily Amarylloideae.

 Thus Traub's Amarylloideae most closely resembles subfamily Amaryllidoideae "sensu" APGIII.
</item 6>

<item 7>
Graph query row:
genus: Cyrtanthus
relationship: PART_OF
taxonomic_connection: Amaryllidoideae
kind: subfamily
</item 7>

<item 8>
Hippeastrinae
Hippeastrinae is a subtribe of plants classified under the tribe Hippeastreae.

 It belongs to the subfamily Amaryllidoideae of the Amaryllis family (Amaryllidaceae).
</item 8>

<item 9>
Ammocharis
Ammocharis is a small genus from sub-Saharan Africa, in the Amaryllidaceae family (subfamily Amaryllidoideae) which includes 7 species distributed in Africa.

 The plant grows as a succulent, above-ground bulb, preferring seasonally wet, hot, sandy soils and full sun.
</item 9>

<item 10>
Aspidistra
Aspidistra is a genus of flowering plants in the family Asparagaceae, subfamily Nolinoideae, native to eastern and southeastern Asia, particularly China and
Vietnam.

 They grow in shade under trees and shrubs.

 Their leaves arise more or less directly from ground level, where their flowers also appear.

 The number of species known has increased considerably from the 1980s onwards, with around 100 accepted as of July 2013 . "

Aspidistra elatior" is common worldwide as a foliage house plant that is very tolerant of neglect.

 It and other species can also be grown in shade outside, where they are generally hardy to -5 C .
</item 10>
=== END ITEM id=a1fce7d83a11f849 ===

=== ITEM id=726361f34f0f5821 ===
Gold sentences and facts:
Sentence 0: Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist, pianist, and primary songwriter for the Christian rock band Relient K.
  Fact 0.0: Matthew Arnold 'Matt' Thiessen was born August 12, 1980.
  Fact 0.1: Matt Thiessen is a Canadian-American musician, singer, and songwriter.
  Fact 0.2: Matt Thiessen is known for being co-founder of the Christian rock band Relient K.
  Fact 0.3: Matt Thiessen is known for being lead singer of the Christian rock band Relient K.
  Fact 0.4: Matt Thiessen is known for being guitarist of the Christian rock band Relient K.
  Fact 0.5: Matt Thiessen is known for being pianist of the Christian rock band Relient K.
  Fact 0.6: Matt Thiessen is known for being primary songwriter for the Christian rock band Relient K.
Sentence 1: George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.
  Fact 1.0: George Thorogood was born February 24, 1950.
  Fact 1.1: George Thorogood is an American musician, singer, and songwriter.
  Fact 1.2: George Thorogood is from Wilmington, Delaware.

Retrieved items:
<item 1>
Better Than the Rest
Better Than the Rest is the third album (mini-album) of songs by George Thorogood and the Destroyers, recorded in 1974 and released in 1979.

 At the time the songs were recorded, Thorogood was an unknown artist who was not signed to a record label.

 In 1979, after Thorogood had released his first two official albums on Rounder Records, "Better Than The Rest" was released by MCA.

 In 1986, the songs from this album were released on compact disc with the title "Nadine" using a different track order.
</item 1>

<item 2>
Bob Thomas (actor)
Bob Thomas (born March 1, 1954 in Appalachia, Virginia ) is a radio personality, actor, and writer.

 He was one of the top radio announcers in Knoxville, Tennessee for 25 years.

 As an actor, he has appeared in many films and hundreds of commercials.

 He wrote two episodes of the popular TV show "Lizzie McGuire".

 In 2004, his voice was heard as "Slammin' Sammy" in the film "Friday Night Lights".
</item 2>

<item 3>
Mike Mattison
Mike Mattison is an American musician and vocalist of the Grammy Award-winning blues rock/soul group, the Tedeschi Trucks Band, as well as lead vocalist and
co-founder of the blues rock trio Scrapomatic.

 Mattison's vocal sound has been described as "strong," with an "expressive blues voice".

 As lead vocalist of Scrapomatic, he picked up a nomination for Minnesota Music Awards best male vocalist, and both he and co-founder Paul Olsen were also
nominated for best R&B Group.

 Mattison was previously the lead vocalist of the Grammy Award winning Derek Trucks Band and has been a main songwriter of all three bands.

 He is also an active essayist who publishes on music and poetry.

 Since 2013 Mattison and Ernest Suarez have edited “Hot Rocks: Songs and Verse,” an ongoing feature in Five Points: A Journal of Literature and Art.
</item 3>

<item 4>
The J. Geils Band
The J. Geils Band was an American rock band formed in 1968, in Worcester, Massachusetts, under the leadership of guitarist John "J." Geils.

 The other band members included vocalist Peter Wolf, harmonica player Richard "Magic Dick" Salwitz, drummer Stephen Bladd, vocalist/keyboardist Seth Justman
and bassist Danny Klein.

 Wolf and Justman served as principal songwriters.

 The band played R&B-influenced blues rock during the 1970s and soon achieved commercial success before moving towards a more mainstream radio-friendly sound in
the early 1980s, which brought the band to its commercial peak.

 After Wolf left the band in 1983 to pursue a solo career, the band released one more album in 1984 with Justman on lead vocals before breaking up in 1985.

 Beginning in 1999, the band had several reunions prior to the death of its namesake founder J. Geils on April 11, 2017.
</item 4>

<item 5>
Matt Thiessen
Matthew Arnold "Matt" Thiessen (born August 12, 1980) is a Canadian-American musician, singer and songwriter known for being co-founder, lead singer, guitarist,
pianist, and primary songwriter for the Christian rock band Relient K.

 With Relient K, he has released eight full-length albums, including three that were certified Gold, and three that peaked in the top twenty on the "Billboard"
200.

 Outside of his work with Relient K, Thiessen maintains a low-profile side project called Matthew Thiessen and the Earthquakes.

 In 2009, he co-produced and collaborated on Owl City's album "Ocean Eyes".
</item 5>

<item 6>
Tom Johnston (musician)
Charles Thomas "Tom" Johnston (born August 15, 1948) is an American musician.

 He is a guitarist and vocalist, known principally as a founder, guitarist, lead vocalist and songwriter for the rock group The Doobie Brothers, as well as for
his own solo career.

 His off-and-on career with The Doobie Brothers spans more than 40 years, during which time the band exhibited numerous successful styles.
</item 6>

<item 7>
George Thorogood and the Destroyers (album)
George Thorogood and the Destroyers is the self-titled debut album by American blues rock band George Thorogood and the Destroyers, released in 1977.

 Consisting mostly of covers of blues hits, it included a medley of John Lee Hooker's "House Rent Boogie" and "One Bourbon, One Scotch, One Beer", the latter a
song written by Rudy Toombs for Amos Milburn, and later covered by Hooker.
</item 7>

<item 8>
George Thorogood
George Thorogood (born February 24, 1950) is an American musician, singer and songwriter from Wilmington, Delaware.

 His "high-energy boogie-blues" sound became a staple of 1980s rock radio, with hits like his original songs "Bad to the Bone" and "I Drink Alone".

 He has also helped popularize older songs by American icons, such as "Move It on Over", "Who Do You Love?

", and "House Rent Boogie/One Bourbon, One Scotch, One Beer", which became staples of classic rock radio.
</item 8>

<item 9>
2120 South Michigan Ave.
2120 South Michigan Ave. is the fifteenth studio album by George Thorogood and the Destroyers.

 It was released on June 14, 2011 on the Capitol Records label.

 The album peaked at #2 on the "Billboard" Top Blues Albums chart.

 The title refers to the address of the offices and recording studios of Chess Records in Chicago.

 The album contains ten covers of songs recorded on Chess Records by artists such as Chuck Berry, Bo Diddley, Willie Dixon, and Muddy Waters; plus a cover of
The Rolling Stones' instrumental "2120 South Michigan Avenue" and two original songs about Chess Records artists.

 Capitol Records approached Thorogood with the idea for the album and selected most of the songs.

 Buddy Guy and Charlie Musselwhite perform on the album, although their work was added after primary recording was complete.
</item 9>

<item 10>
More George Thorogood and The Destroyers
More George Thorogood and the Destroyers is the fourth album by George Thorogood and The Destroyers, released in 1980.

 An alternate name for the album is I'm Wanted.
</item 10>
=== END ITEM id=726361f34f0f5821 ===

=== ITEM id=17f61c86b0e9c1eb ===
Gold sentences and facts:
Sentence 0:  He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth Games in Edinburgh, Scotland.
  Fact 0.0: He also participated in the 10,000 metres at the 1967 Pan American Games.
  Fact 0.1: The 1967 Pan American Games were held in Winnipeg, Manitoba, Canada.
  Fact 0.2: He represented Guyana at the 1970 British Commonwealth Games.
  Fact 0.3: The 1970 British Commonwealth Games were held in Edinburgh, Scotland.
Sentence 1:  The race consists of 25 laps around an Olympic-sized track.
  Fact 1.0: The race consists of 25 laps.
  Fact 1.1: The 25 laps are around an Olympic-sized track.

Retrieved items:
<item 1>
Siegmar Ohlemann
Siegmar ("Sig") Ohlemann (born May 11, 1938) is a retired middle distance runner, who represented Canada in three events (800 metres, 4 × 100 metres and 4 × 400
metres) at the 1960 Summer Olympics in Rome, Italy.

 He claimed the silver medal in 800 metres at the 1963 Pan American Games in Brazil, behind fellow Canadian Don Bertoia.
</item 1>

<item 2>
Nora Rocha
Nora Leticia Rocha de la Cruz (born December 18, 1967 in Monclova, Coahuila) is a retired female track and field athlete from Mexico, who competed in the 5000
and 10,000 metres.

 She claimed the gold medal in the women's 10,000 metres at the 1999 Pan American Games in Winnipeg, Canada.
</item 2>

<item 3>
Dave Bailey (athlete)
David Bailey (born March 17, 1945 in Toronto, Ontario) is a retired track and field athlete, who represented Canada at the 1968 Summer Olympics in the men's
1.500 metres.

 He was the first Canadian to run the mile in less than 4 minutes (3:59.1) in San Diego, CA on June 11, 1966 and the first Canadian to the run the mile in less
than 4 minutes in Canada (3:57.7)in Toronto on July 22, 1967.

 A resident of Willowdale, Ontario he won the bronze medal in this event at the 1967 Pan American Games and the silver medal in this event at the 1968 World
University Games He was two-time Canadian Universities Track and Field Athlete of Year (1965, 1967) and two-time inductee into the University of Toronto Sports
Hall of Fame (individually 1998 and team 2003).
</item 3>

<item 4>
Brian MacLaren
Brian MacLaren (born 21 December 1943) is a Canadian sprinter.

 He competed in the men's 400 metres at the 1972 Summer Olympics.

 He finished second in the 1967 Pan American Games 4 × 400 metres relay (with Bill Crothers, Ross MacKenzie, and Robert McLaren) and third in the 1967 Pan
American Games 800 metres.

 He won a silver medal in the 1966 British Empire and Commonwealth Games 4 x 440 yards relay with Don Domansky, Ross MacKenzie and Bill Crothers.
</item 4>

<item 5>
John Lindsay (Paralympian)
That year, he had a Victorian Institute of Sport scholarship.

 He was also working as a fitness instructor in 1992, held world records in the 100 m and 200 m events, and was ranked 6th in the world in the 400 m.

 He won a gold medal in the men's athletics 100 m T52 event at the 1996 Summer Paralympics with a time of 15.22, a silver medal in the 200 m T52 event with a
time of 27.38, and a bronze medal in the 400 m T52 event with a time of 52.93.

 At the 2000 Sydney Games, he won a gold medal in the Men's 100 m T53 event, a silver medal as part of the Men's 4x100 m Relay T54 team, and a bronze medal in
the Men's 200 m T53 event; he was also part of the Men's 4x400 m Relay T54 team, which was the only one to qualify in its heat, but it did not make it to the
finals.
</item 5>

<item 6>
John Lindsay (Paralympian)
At the 2004 Athens Games, he came seventh in the first round of the Men's 100 m T53 event and sixth in the third round of the Men's 200 m T53 event.

 He was an Australian Institute of Sport scholarship holder in 1995 and 2000.
</item 6>

<item 7>
Harry Prowell
Harry Prowell A.A.(10 July 1936 – 27 June 2000) was a Guyanese long distance runner who represented Guyana in the Marathon at the 1968 Summer Olympics in Mexico
City, Mexico.

 He is known to be one of the greatest Marathon runners Guyana has ever produced, setting the national record in 1968.

 To date, he is the only Guyanese ever to compete in the Marathon at the Summer Olympic Games and one of the most prominent Indo-Caribbean long distance runners
of his time.

 He also participated in the 10,000 metres at the 1967 Pan American Games in Winnipeg, Manitoba, Canada and represented Guyana at the 1970 British Commonwealth
Games in Edinburgh, Scotland.
</item 7>

<item 8>
European 10,000m Cup
The European 10,000m Cup is an annual 10,000 metres race for European athletes which was first held in 1997.

 The competition is organised by the European Athletics Association and first began as the European 10000 Metres Challenge after the event was removed from the
European Cup programme.

 The competition has roots in the Iberian 10,000 metres Championships – a competition between Spanish and Portuguese athletes that was held between 1991 and
1996 – and the first five editions of the European 10000 Metres Challenge were held in the Iberian Peninsula.

 The event was first held under its current title in 2005.
</item 8>

<item 9>
Dave Bailey (athlete)
He was a member of 9 Canadian National Track and Field Teams competing at World University Games, Budapest 1965 and Tokyo 1967, Commonwealth Games, Kingston,
Jamaica, 1966, Pan American Games, Winnipeg, 1967, Commonwealth vs USA, Los Angeles, 1967, Pre-Olympic Games, Mexico City, 1967, Soviet Union Tour, 1968,
European Tour, 1968, Olympic Games, Mexico City, 1968.
</item 9>

<item 10>
10,000 metres
The 10,000 metres or 10,000-meter run is a common long-distance track running event.

 The event is part of the athletics programme at the Olympic Games and the World Championships in Athletics and is common at championship level events.

 The race consists of 25 laps around an Olympic-sized track.

 It is less commonly held at track and field meetings, due to its duration.

 The 10,000 metres track race is usually distinguished from its road running counterpart, the 10K run, by its reference to the distance in metres rather than
kilometres.
</item 10>
=== END ITEM id=17f61c86b0e9c1eb ===
