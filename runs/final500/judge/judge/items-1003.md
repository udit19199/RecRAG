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

This file has 3 items.

=== ITEM id=e8b41c4917ee6de0 ===
Gold sentences and facts:
Sentence 0: R.H. Thomson as Matthew Cuthbert
  Fact 0.0: R.H. Thomson played the role of Matthew Cuthbert.

Retrieved items:
<item 1>
Anne (TV series)
of seasons No. of episodes 7 ( list of episodes ) Production Executive producer ( s ) Elizabeth Bradley Alex Sapot Sally Catto Debra Hayward Alison Owen Miranda
de Pencier Moira Walley - Beckett Producer ( s ) Susan Murdoch Cinematography Bobby Shore Running time 44 minutes Production company ( s ) Pelican Ballet
Northwood Entertainment Distributor Netflix Release Original network CBC Television Netflix ( worldwide ) Picture format 4K ( Ultra HD ) Audio format Dolby
Digital 5.1 with Descriptive Video Service track Original release March 19 , 2017 ( 2017 - 03 - 19 ) -- present ( present ) External links Website Anne ( titled
Anne with an E for international distribution on Netflix ) is a Canadian drama television series based on the 1908 novel Anne of Green Gables by Lucy Maud
Montgomery , and adapted by Emmy Award - winning writer and producer Moira Walley - Beckett . It airs on CBC in Canada , and elsewhere in the world it is
available for streaming on Netflix . The first season consists of seven episodes , with Niki Caro directing the 90 - minute season premiere .
</item 1>

<item 2>
Anne (TV series)
7 `` Wherever You Are Is My Home '' Amanda Tapping Moira Walley - Beckett April 30 , 2017 ( 2017 - 04 - 30 ) N / A The Cuthberts vow to do whatever it takes to
save the farm which reminds Anne of the strength of friendship and love . Reception ( edit ) Critical response ( edit ) The series has achieved a rating of 88 %
on Rotten Tomatoes , the site 's critical consensus stating : `` Anne with an E uses its complex central character to offer a boldly stylish , emotionally
resonant spin on classic source material that satisfies in its own right . '' The series has received a rating of 79 on Metacritic based on fifteen reviews ,
indicating `` generally favorable reviews '' . Emily Ashby , writing for Common Sense Media , calls the series an `` exceptional '' and `` spectacular ''
interpretation , giving it four out of five stars . Tasha Cerny , contributor for the Tracking Board , praises the cinematography as lush and colourful , the
characters vibrant , and the plot `` surprisingly thrilling for a story about a young girl living in a small secluded community in the late nineteenth century .
</item 2>

<item 3>
*Anne* (internationally titled *Anne with an E*) is a Canadian drama adapted by Moira Walley-Beckett from Lucy Maud Montgomery’s 1908 novel *Anne of Green
Gables*. It premiered on CBC on March 19, 2017, and streams internationally on Netflix. Its first season has seven episodes; Niki Caro directed the 90-minute
premiere. Episodes run about 44 minutes, with 4K video and Dolby Digital 5.1
</item 3>

<item 4>
*Anne* (*Anne with an E*), Moira Walley-Beckett’s CBC/Netflix adaptation of L.M. Montgomery’s *Anne of Green Gables*, premiered in March 2017. Amybeth McNulty
stars as orphan Anne, mistakenly sent to siblings Matthew and Marilla Cuthbert. Critics praised its cast, visuals, emotional depth, and attention to historical
hardship, but some objected to its darker tone, added tragedy, melodrama, and heavy-handed feminist changes, arguing
</item 4>

<item 5>
Anne (TV series)
When Matthew goes to pick the child up at the railway station , he finds not a boy , but a high - spirited and talkative girl , Anne Shirley . At first , the
Cuthberts are inclined to send her back , particularly after Marilla 's brooch goes missing , and Anne , in despair , runs away . Reconciled and settled in her
new home and school , Anne once again displays boundless enthusiasm which is nevertheless easily turned into despair when things go wrong , which they often do
. Slowly , her ebullient nature wins over those around her . Cast ( edit ) Main ( edit ) Amybeth McNulty as Anne Shirley Geraldine James as Marilla Cuthbert
R.H. Thomson as Matthew Cuthbert Dalila Bela as Diana Barry Lucas Jade Zumann as Gilbert Blythe Corrine Koslo as Rachel Lynde Aymeric Jett Montaz as Jerry
Baynard Others ( edit ) Jonathan Holmes as Mr. William Barry Helen Johns as Mrs. Eliza Barry Ryan Kiera Armstrong as Minnie Mary Barry Deborah Grover as
Josephine Barry Wayne Best as John Blythe Phillip Williams as Thomas Lynde David Ingram as Mr.
</item 5>

<item 6>
*Anne* premiered on CBC on March 19, 2017, and its first season ended April 30. It follows orphan Anne Shirley, mistakenly sent to siblings Matthew and Marilla
Cuthbert’s Green Gables farm instead of the boy they expected to help them. Netflix released the series as *Anne with an E* on May 12, 2017. CBC and Netflix
renewed it for a 10-episode second season, which began production in November 2017 and was
</item 6>

<item 7>
The reimagined *Anne of Green Gables* series, *Anne with an E*, was renewed by CBC and Netflix for a second season in August 2017, with episodes written by a
team of women. Coverage highlighted its darker, more feminist take on Anne Shirley. Casting announcements and behind-the-scenes reports followed filming in
Millbrook, Ontario.
</item 7>

<item 8>
*Anne* (also known as *Anne with an E*) is a Canadian English-language drama created and written by Moira Walley-Beckett, based on Lucy Maud Montgomery’s *Anne
of Green Gables*. It stars Amybeth McNulty, Geraldine James, and R.H. Thomson. In episode seven, the Cuthberts fight to save their farm, highlighting friendship
and love. Critics praised its emotional storytelling and visuals; it scored 88% on Rotten Tomatoes
</item 8>

<item 9>
Anne (TV series)
Anne ( TV series ) - wikipedia Anne ( TV series ) Jump to : navigation , search Anne Also known as Anne with an E Genre Drama Created by Moira Walley - Beckett
Based on Anne of Green Gables by Lucy Maud Montgomery Screenplay by Moira Walley - Beckett Starring Amybeth McNulty Geraldine James R.H. Thomson Lucas Jade
Zumann Dalila Bela Corrine Koslo Aymeric Jett Montaz Opening theme `` Ahead by a Century '' by The Tragically Hip Country of origin Canada Original language ( s
) English No. of seasons No.
</item 9>

<item 10>
Anne (TV series)
'' She went on to call Anne Shirley an `` accidental feminist '' , and how she `` really wanted to tell this story now . '' Broadcast and Release ( edit ) The
series initially premiered on March 19 , 2017 , on CBC and aired on a weekly basis , the season finale airing on April 30 , 2017 . The series debuted on Netflix
on May 12 , 2017 , under the title Anne with an E . Episodes ( edit ) Season 1 ( 2017 ) ( edit ) No . Title Directed by Written by Original air date Can .
viewers ( millions ) `` Your Will Shall Decide Your Destiny '' Niki Caro Moira Walley - Beckett March 19 , 2017 ( 2017 - 03 - 19 ) 0.999 When a miscommunication
brings a girl , Anne Shirley , to Green Gables instead of a boy , Marilla and Matthew Cuthbert are faced with a life - changing decision .
</item 10>
=== END ITEM id=e8b41c4917ee6de0 ===

=== ITEM id=62adec49ebbbc8a5 ===
Gold sentences and facts:
Sentence 0: Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher Usmanov ( 30.04 % ) 15 ! $22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M $3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 ! $2.3 B Steel and energy Theatre production Planet Hollywood Ocean Software Huddersfield Town Dean Hoyle 09 ! $475 M Card Factory Leicester City The Srivaddhanaprabha Family 10 ! $4.9 B King Power International Group Liverpool ( more information ) John W. Henry 11 ! $2.5 B Fenway Sports Group Manchester City ( more information ) Mansour bin Zayed Al Nahyan ( 87 % ) China Media Capital ( 13 % ) 01 ! $44 B Oil and industry investment Government Manchester United ( more information ) The Glazer Family 06 ! $4.9 B First Allied Corporation , Tampa Bay Buccaneers Newcastle United ( more information ) Mike Ashley 04 ! $3.3 B Sports Direct International Southampton Gao Jisheng ( 80 % ) Katharina Liebherr ( 20 % ) 07 ! $3.9 B + Lander Sports Development Co Ltd Inheritance Stoke City The Coates Family 08 ! $3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B -- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport , Sunday Sport Gold Group International Straumur Investment Bank
  Fact 0.0: Arsenal's owner E. Stanley Kroenke holds 67.05% of the club.
  Fact 0.1: Arsenal's owner Alisher Usmanov holds 30.04% of the club.
  Fact 0.2: Arsenal's estimated combined owner net worth is $22.4 B.
  Fact 0.3: Arsenal's owners' source of wealth is metal and mining and investment.
  Fact 0.4: Kroenke Sports Enterprises is listed for Arsenal.
  Fact 0.5: Bournemouth's owner Maxim Demin holds 75% of the club.
  Fact 0.6: Bournemouth's owner PEAK6 Investments holds 25% of the club.
  Fact 0.7: Bournemouth's source of wealth is petrochemicals trading.
  Fact 0.8: Seven Developments is listed for Bournemouth.
  Fact 0.9: Brighton & Hove Albion's owner Tony Bloom holds 75.61% of the club.
  Fact 0.10: Brighton & Hove Albion's owner net worth is $1.7 B.
  Fact 0.11: Brighton & Hove Albion's source of wealth is online gambling, real estate, property, land development and investments.
  Fact 0.12: Burnley's owner Mike Garlick holds 49.3% of the club.
  Fact 0.13: Burnley's owner John Banaszkiewicz holds 27.55% of the club.
  Fact 0.14: Burnley's estimated combined owner net worth is $80 M.
  Fact 0.15: Burnley's sources of wealth are Michael Bailey Associates and Freight Investor Services.
  Fact 0.16: Chelsea's owner is Roman Abramovich.
  Fact 0.17: Chelsea's owner net worth is $9.3 B.
  Fact 0.18: Chelsea's source of wealth is oil and industry.
  Fact 0.19: Crystal Palace's owners are Steve Parish, Martin Long, Joshua Harris, David S. Blitzer, Jeremy Hosking and Stephen Browett.
  Fact 0.20: Crystal Palace's owners' net worths are listed as $65 M, $100 M, $3.2 B, $1.3 B, $250 M and $110 M.
  Fact 0.21: Crystal Palace's sources of wealth are listed as Tag Worldwide, Churchill Insurance, Apollo Global Management, Blackstone Group, Marathon Asset Management and Farr Vintners.
  Fact 0.22: Everton's owner Farhad Moshiri holds 49.9% of the club.
  Fact 0.23: Everton's owners include Bill Kenwright CBE and Jon Woods.
  Fact 0.24: Everton's estimated combined owner net worth is $2.3 B.
  Fact 0.25: Everton's sources of wealth are steel and energy, theatre production, Planet Hollywood and Ocean Software.
  Fact 0.26: Huddersfield Town's owner is Dean Hoyle.
  Fact 0.27: Huddersfield Town's owner net worth is $475 M.
  Fact 0.28: Huddersfield Town's source of wealth is Card Factory.
  Fact 0.29: Leicester City's owner is The Srivaddhanaprabha Family.
  Fact 0.30: Leicester City's owner net worth is $4.9 B.
  Fact 0.31: Leicester City's source of wealth is King Power International Group.
  Fact 0.32: Liverpool's owner is John W. Henry.
  Fact 0.33: Liverpool's owner net worth is $2.5 B.
  Fact 0.34: Liverpool's source of wealth is Fenway Sports Group.
  Fact 0.35: Manchester City's owner Mansour bin Zayed Al Nahyan holds 87% of the club.
  Fact 0.36: Manchester City's owner China Media Capital holds 13% of the club.
  Fact 0.37: Manchester City's owner net worth is $44 B.
  Fact 0.38: Manchester City's sources of wealth are oil and industry investment and government.
  Fact 0.39: Manchester United's owner is The Glazer Family.
  Fact 0.40: Manchester United's owner net worth is $4.9 B.
  Fact 0.41: Manchester United's sources of wealth are First Allied Corporation and Tampa Bay Buccaneers.
  Fact 0.42: Newcastle United's owner is Mike Ashley.
  Fact 0.43: Newcastle United's owner net worth is $3.3 B.
  Fact 0.44: Newcastle United's source of wealth is Sports Direct International.
  Fact 0.45: Southampton's owner Gao Jisheng holds 80% of the club.
  Fact 0.46: Southampton's owner Katharina Liebherr holds 20% of the club.
  Fact 0.47: Southampton's owner net worth is $3.9 B.
  Fact 0.48: Southampton's source of wealth is Lander Sports Development Co Ltd and inheritance.
  Fact 0.49: Stoke City's owner is The Coates Family.
  Fact 0.50: Stoke City's owner net worth is $3.1 B.
  Fact 0.51: Stoke City's source of wealth is Bet365.
  Fact 0.52: Swansea City's owners Stephen Kaplan and Jason Levien hold 68% of the club.
  Fact 0.53: Swansea City's Supporters Trust holds 21.1% of the club.
  Fact 0.54: Swansea City's sources of wealth are Memphis Grizzlies, D.C. United and investment trust.
  Fact 0.55: Tottenham Hotspur's owner Joe Lewis holds 70.6% of the 85% holding company.
  Fact 0.56: Tottenham Hotspur's owner Daniel Levy holds 29.4% of the 85% holding company.
  Fact 0.57: Tottenham Hotspur's owner net worth is $5.6 B.
  Fact 0.58: Tottenham Hotspur's sources of wealth are Tavistock Group and ENIC Group.
  Fact 0.59: Watford's owner is Gino Pozzo.
  Fact 0.60: Watford's owner is described as a businessman.
  Fact 0.61: West Bromwich Albion's owner Guochuan Lai holds 87.8% of the club.
  Fact 0.62: West Bromwich Albion's owner net worth is $6.2 B.
  Fact 0.63: West Bromwich Albion's source of wealth is investment.
  Fact 0.64: West Ham United's owner David Sullivan holds 51.1% of the club.
  Fact 0.65: West Ham United's owner David Gold holds 35.1% of the club.
  Fact 0.66: West Ham United's owner Straumur Investment Bank holds 10% of the club.
  Fact 0.67: West Ham United's owners' net worths are listed as $520 M and $325 M.
  Fact 0.68: West Ham United's sources of wealth are Daily Sport, Sunday Sport, Gold Group International and Straumur Investment Bank.
Sentence 1: Club Owner ( s ) Estimated combined net worth ( £ m ) Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % ) Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % ) Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company Middlesbrough Steve Gibson 10 ! $260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn - Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C. Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 ! $155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment
  Fact 1.0: Aston Villa's owner is Dr. Tony Xia.
  Fact 1.1: Aston Villa's owner net worth is $1.2 B.
  Fact 1.2: Aston Villa's source of wealth is Recon Group.
  Fact 1.3: Barnsley's owner is Patrick Cryne.
  Fact 1.4: Barnsley's owner net worth is $40 M.
  Fact 1.5: Barnsley's source of wealth is iSOFT.
  Fact 1.6: Birmingham City's owners are Paul Suen (60.78%), Wang Lei (5.28%) and Carson Yeung (6.51%).
  Fact 1.7: Birmingham City's estimated combined owner net worth is $570 M.
  Fact 1.8: Birmingham City's source of wealth is Birmingham International Holdings.
  Fact 1.9: Bolton Wanderers' owner is Eddie Davies CBE.
  Fact 1.10: Bolton Wanderers' owner net worth is $52 M.
  Fact 1.11: Bolton Wanderers' source of wealth is Strix.
  Fact 1.12: Brentford's owner Matthew Benham holds 96% of the club.
  Fact 1.13: Bristol City's owner is Stephen Lansdown.
  Fact 1.14: Bristol City's owner net worth is $1.9 B.
  Fact 1.15: Bristol City's source of wealth is Hargreaves Lansdown.
  Fact 1.16: Burton Albion's owner is Ben Robinson.
  Fact 1.17: Cardiff City's owner Vincent Tan holds 51% of the club.
  Fact 1.18: Cardiff City's owner net worth is $820 M.
  Fact 1.19: Cardiff City's source of wealth is Berjaya Group.
  Fact 1.20: Derby County's owner is Mel Morris.
  Fact 1.21: Derby County's owner net worth is $520 M.
  Fact 1.22: Fulham's owner is Shahid Khan.
  Fact 1.23: Fulham's owner net worth is $8.9 B.
  Fact 1.24: Fulham's source of wealth is Flex-N-Gate Corp, an automobile parts manufacturer.
  Fact 1.25: Hull City's owner is The Allam Family.
  Fact 1.26: Hull City's owner net worth is $325 M.
  Fact 1.27: Hull City's source of wealth is Allam Marine.
  Fact 1.28: Ipswich Town's owner is Marcus Evans.
  Fact 1.29: Ipswich Town's owner net worth is $825 M.
  Fact 1.30: Ipswich Town's source of wealth is Marcus Evans Group.
  Fact 1.31: Leeds United's owners are Andrea Radrizzani, Eleonora Sports Ltd and Aser Group Company.
  Fact 1.32: Middlesbrough's owner is Steve Gibson.
  Fact 1.33: Middlesbrough's owner net worth is $260 M.
  Fact 1.34: Middlesbrough's source of wealth is Bulkhaul Limited.
  Fact 1.35: Millwall's owners are John Berylson and Richard Smith.
  Fact 1.36: Millwall's owner net worth is $175 M.
  Fact 1.37: Millwall's source of wealth is Chestnut Hill Ventures.
  Fact 1.38: Norwich City's owners Delia Smith and Michael Wynn-Jones hold 53% of the club.
  Fact 1.39: Norwich City's owner Michael Foulger holds 15% of the club.
  Fact 1.40: Norwich City's owner net worth is $30 M.
  Fact 1.41: Norwich City's sources of wealth are food industry, publishing and poultry.
  Fact 1.42: Nottingham Forest's owner is Evangelos Marinakis.
  Fact 1.43: Nottingham Forest's owner net worth is $650 M.
  Fact 1.44: Nottingham Forest's owner is also owner of Olympiacos F.C.
  Fact 1.45: Preston North End's owner Trevor Hemmings CVO holds 51.4% of the club.
  Fact 1.46: Preston North End's owner net worth is $720 M.
  Fact 1.47: Preston North End's sources of wealth are Blackpool Tower, horse racing, investment, Northern Trust and Sportech.
  Fact 1.48: Queens Park Rangers' owner Tony Fernandes holds 66% of the club.
  Fact 1.49: Queens Park Rangers' owner Lakshmi Mittal holds 33% of the club.
  Fact 1.50: Queens Park Rangers' owners' net worths are listed as $345 M and $16.4 B.
  Fact 1.51: Queens Park Rangers' sources of wealth are Air Asia, Tune Group and ArcelorMittal.
  Fact 1.52: Reading's owners are Narin Niruttinanon (50%), Sasima Srivikorn (25%) and Sumrith Thanakarnjanasuth (25%).
  Fact 1.53: Sheffield United's owner Abdullah bin Musa'ed holds 50% of the club.
  Fact 1.54: Sheffield United's owner net worth is $155 M.
  Fact 1.55: Sheffield United's sources of wealth are property and oil.
  Fact 1.56: Sheffield Wednesday's owner is Dejphon Chansiri.
  Fact 1.57: Sheffield Wednesday's source of wealth is Thai Union Group.
  Fact 1.58: Sunderland's owner is Ellis Short.
  Fact 1.59: Sunderland's owner net worth is $2.9 B.
  Fact 1.60: Sunderland's sources of wealth are Lone Star Funds and Skibo Castle.
  Fact 1.61: Wolverhampton Wanderers' owners are Fosun International (parent company) and Guo Guangchang (co-owner of Fosun).
  Fact 1.62: Wolverhampton Wanderers' owner net worth is $6.2 B.
  Fact 1.63: Wolverhampton Wanderers' source of wealth is investment.

Retrieved items:
<item 1>
List of owners of English football clubs
Tony Xia $1.2 B ! $1.2 B Recon Group Barnsley Patrick Cryne 14 ! $40 M iSOFT Birmingham City ( more information ) Paul Suen ( 60.78 % ) Wang Lei ( 5.28 % )
Carson Yeung ( 6.51 % ) 21 ! $570 M Birmingham International Holdings Bolton Wanderers Eddie Davies CBE 13 ! $52 M Strix Brentford Matthew Benham ( 96 % )
Bristol City Stephen Lansdown 01 ! $1.9 B Hargreaves Lansdown Burton Albion Ben Robinson Cardiff City Vincent Tan ( 51 % ) 01 ! $820 M Berjaya Group Derby
County Mel Morris $520 M King Fulham ( more information ) Shahid Khan 09 ! $8.9 B Flex - N-Gate Corp ( automobile parts manufacturer ) Hull City The Allam
Family 15 ! $325 M Allam Marine Ipswich Town Marcus Evans 02 ! $825 M Marcus Evans Group Leeds United Andrea Radrizzani Eleonora Sports Ltd Aser Group Company
Middlesbrough Steve Gibson 10 !
</item 1>

<item 2>
List of owners of English football clubs
$250 M DW Sports Fitness , JJB Sports League Two ( edit ) Club Owner ( s ) Estimated combined net worth Source of wealth Accrington Stanley The Styring Family (
12 % ) Accrington Stanley Supporters Fund ( 12 % ) Peter Marsden ( 10 % ) Investment trust Barnet Blackpool Owen Oyston ( 80 % ) Valeri Belokon ( 25 % ) 07 !
$105 M $195 M Publishing , Media & Estate Agency Finance , Media , Food Industry & Mining Cambridge United Grosvenor Fund Management Grosvenor Group Owned by
the Duke of Westminster and family Property Carlisle United ( more information ) A. Jenkins ( 37 % of 93 % holding company ) J. Nixon ( 18 % ) S. Pattison ( 18
% ) The United Trust ( 25.4 % ) Pioneer Foods Businessman ( retired ) Carlisle Glass Investment Trust Cheltenham Town Paul Baker Chesterfield Dave Allen
Colchester United Robbie Cowling 09 !
</item 2>

<item 3>
List of owners of English football clubs
$22.4 B Kroenke Sports Enterprises Metal and mining , investment Bournemouth Maxim Demin ( 75 % ) PEAK6 Investments ( 25 % ) Petrochemicals trading Seven
Developments Brighton & Hove Albion Tony Bloom ( 75.61 % ) 12 ! $1.7 B Online Gambling , Real Estate , Property , Land Development & Investments Burnley Mike
Garlick ( 49.3 % ) John Banaszkiewicz ( 27.55 % ) 18 ! $80 M -- - Michael Bailey Associates Freight Investor Services Chelsea ( more information ) Roman
Abramovich 03 ! $9.3 B Oil and industry Crystal Palace Steve Parish Martin Long Joshua Harris David S. Blitzer Jeremy Hosking Stephen Browett 15 ! $65 M $100 M
$3.2 B $1.3 B $250 M $110 M Tag Worldwide Churchill Insurance Apollo Global Management Blackstone Group Marathon Asset Management Farr Vintners Everton ( more
information ) Farhad Moshiri ( 49.9 % ) Bill Kenwright CBE Jon Woods 15 !
</item 3>

<item 4>
List of owners of English football clubs
Jump up ^ `` Shahid Khan '' . Forbes . Forbes.com LLC. 19 May 2017 . Retrieved 19 May 2017 . Jump up ^ `` Who 's Who At The KC '' . Hull City F.C. 25 June 2012
. Retrieved 9 May 2014 . Jump up ^ `` Sunday Times Rich List 2017 reveals Yorkshire 's 20 wealthiest people '' . Hull Daily Mail . Archived from the original on
2 May 2017 . Retrieved 19 May 2017 . Jump up ^ www.calcioefinanza.it Leeds http://www.calcioefinanza.it/2017/01/04/leeds-cellino-radrizzani/ Leeds Check url =
value ( help ) . Retrieved 2017 - 01 - 10 . Missing or empty title = ( help ) Jump up ^ leeds united
https://www.leedsunited.com/news/club-news/21921/club-statement . Retrieved 2017 - 05 - 23 .
</item 4>

<item 5>
List of owners of English football clubs
List of owners of English football clubs - wikipedia List of owners of English football clubs Jump to : navigation , search This is a list of the current owners
of English football clubs , as well as ( in some cases ) their estimated net worth and source of wealth . All figures are listed in millions rather than
billions . For example , £ 1,000 m would be one billion ( 1,000,000,000 ) pounds . Only shareholders with a significant interest ( above 10 % ) are listed .
Contents ( hide ) 1 Premier League 2 Football League Championship 3 League One 4 League Two 5 References Premier League ( edit ) This list is incomplete ; you
can help by expanding it . Club Owner ( s ) Estimated combined net worth Source of wealth Arsenal ( more information ) E. Stanley Kroenke ( 67.05 % ) Alisher
Usmanov ( 30.04 % ) 15 !
</item 5>

<item 6>
List of owners of English football clubs
$155 M Property Oil Sheffield Wednesday Dejphon Chansiri Thai Union Group Sunderland Ellis Short 09 ! $2.9 B Lone Star Funds , Skibo Castle Wolverhampton
Wanderers Fosun International ( parent company ) Guo Guangchang ( co-owner of Fosun ) $6.2 B Investment League One ( edit ) Club Owner ( s ) Estimated combined
net worth Source of wealth AFC Wimbledon Wimbledon Football Club Supporters ' Society ( The Dons Trust ) Supporters Trust Blackburn Rovers Balaji & Venkatesh
Rao ( 99.9 % ) 04 ! $650 M Venkateshwara Hatcheries Group Bradford City Bristol Rovers Dwane Sports ( 92 % ) Banking and tourism Bury Save Our Shakers Trust (
63.8 % ) The Bury F.C. Supporters Society Ltd ( Forever Bury ) ( 11 % ) Investment trusts Charlton Athletic Roland Duchatelet ( 95 % ) 24 ! Staprix NV Doncaster
Rovers John Ryan Dick Watson ( 26 % ) Terry Bramall CBE 07 !
</item 6>

<item 7>
List of owners of English football clubs
$3.1 B Bet365 Swansea City Stephen Kaplan & Jason Levien ( 68 % ) Swansea City Supporters Trust ( 21.1 % ) -- - -- - Memphis Grizzlies and D.C. United
Investment trust Tottenham Hotspur ( more information ) Joe Lewis ( 70.6 % of 85 % holding company ) Daniel Levy ( 29.4 % of 85 % holding company ) 05 ! $5.6 B
-- - Tavistock Group ENIC Group Watford Gino Pozzo -- - Businessman West Bromwich Albion ( more information ) Guochuan Lai ( 87.8 % ) 18 ! $6.2 B Investment
West Ham United ( more information ) David Sullivan ( 51.1 % ) David Gold ( 35.1 % ) Straumur Investment Bank ( 10 % ) 13 ! $520 M $325 M -- - Daily Sport ,
Sunday Sport Gold Group International Straumur Investment Bank Football League Championship ( edit ) Club Owner ( s ) Estimated combined net worth ( £ m )
Source of wealth Aston Villa ( more information ) Dr. Tony Xia $1.2 B !
</item 7>

<item 8>
List of owners of English football clubs
$260 M Bulkhaul Limited Millwall John Berylson Richard Smith 11 ! $175 M Chestnut Hill Ventures Norwich City ( more information ) Delia Smith and Michael Wynn -
Jones ( 53 % ) Michael Foulger ( 15 % ) 20 ! $30 M Food industry Publishing Poultry Nottingham Forest Evangelos Marinakis 21 ! $650 M Owner of Olympiacos F.C.
Preston North End Trevor Hemmings CVO ( 51.4 % ) 02 ! $720 M Blackpool Tower , horse racing , investment , Northern Trust , Sportech Queens Park Rangers ( more
information ) Tony Fernandes ( 66 % ) Lakshmi Mittal ( 33 % ) 03 ! $345 M $16.4 B Air Asia , Tune Group ArcelorMittal Reading ( more information ) Narin
Niruttinanon ( 50 % ) Sasima Srivikorn ( 25 % ) Sumrith Thanakarnjanasuth ( 25 % ) Sheffield United Abdullah bin Musa'ed ( 50 % ) 03 !
</item 8>

<item 9>
List of owners of English football clubs
Brentford F.C . Retrieved 29 May 2014 . Jump up ^ `` # 1098 Stephen Lansdown '' . Forbes . Retrieved 19 May 2017 . Jump up ^ `` Club Ownership '' . Cardiff City
F.C . Retrieved 30 November 2013 . Jump up ^ `` Forbes ' Billionaires '' . Forbes . Jump up ^ `` Club Statement : Mel Morris Completes Purchase Of Derby County
'' . Derby County F.C. 4 September 2015 . Retrieved 4 September 2015 . Jump up ^ `` Sunday Times Rich List : Candy Crush tycoon Mel Morris and Duke of
Devonshire among Derbyshire 's millionaires '' . Derby Telegraph . 26 April 2015 . Archived from the original on 3 May 2015 . Retrieved 26 April 2015 . Jump up
^ `` Welcome To Shahid Khan '' . Fulham FC. 12 July 2013 . Retrieved 12 July 2013 . Jump up ^ `` Shahid Khan '' . Forbes .
</item 9>

<item 10>
List of owners of English football clubs
`` Trevor Hemmings ' Northern Trust returns to profit '' . The Daily Telegraph . London . Retrieved 12 February 2013 . Jump up ^ `` Company details '' . Queens
Park Rangers F.C. 10 July 2012 . Archived from the original on 10 August 2012 . Retrieved 12 February 2012 . Jump up ^ `` # 28 Anthony Fernandes '' . Forbes .
Retrieved 27 May 2014 . Jump up ^ `` # 56 Lakshmi Mittal '' . Forbes . Retrieved 19 May 2017 . Jump up ^ Details , Company . `` COMPANY DETAILS '' . Retrieved
29 July 2016 . Jump up ^ `` Fantasy football nut Saudi Prince Abdullah is joint owner of League One side Sheffield United '' . Daily Mail . London . 3 September
2013 . Retrieved 5 September 2013 . Jump up ^ `` Board & Management '' . A.F.C. Sunderland . Retrieved 9 May 2014 .
</item 10>
=== END ITEM id=62adec49ebbbc8a5 ===

=== ITEM id=602cd316b268d682 ===
Gold sentences and facts:
Sentence 0: Third Generation ( GK Series ) Overview Production 2013 -- present Model years 2014 -- present Assembly Yorii , Japan Sumaré , Brazil Ayutthaya , Thailand Pingtung , Taiwan Karawang , Indonesia Malacca , Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 - speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in ) Height 1,525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid )
  Fact 0.0: The Third Generation (GK Series) has production from 2013 to the present.
  Fact 0.1: The Third Generation (GK Series) has model years from 2014 to the present.
  Fact 0.2: The Third Generation (GK Series) is assembled in Yorii, Japan.
  Fact 0.3: The Third Generation (GK Series) is assembled in Sumaré, Brazil.
  Fact 0.4: The Third Generation (GK Series) is assembled in Ayutthaya, Thailand.
  Fact 0.5: The Third Generation (GK Series) is assembled in Pingtung, Taiwan.
  Fact 0.6: The Third Generation (GK Series) is assembled in Karawang, Indonesia.
  Fact 0.7: The Third Generation (GK Series) is assembled in Malacca, Malaysia.
  Fact 0.8: The Third Generation (GK Series) is assembled in Guangzhou, China (Guangqi Honda).
  Fact 0.9: The Third Generation (GK Series) is assembled in Celaya, Guanajuato, Mexico, from February 2014.
  Fact 0.10: The Third Generation (GK Series) is assembled in Tapukara, India, from January 2015.
  Fact 0.11: The Third Generation (GK Series) was designed by Riku Wada in 2010.
  Fact 0.12: The Third Generation (GK Series) is related to the Honda City.
  Fact 0.13: The Third Generation (GK Series) is related to the Honda HR-V.
  Fact 0.14: The Third Generation (GK Series) is related to the Honda Freed.
  Fact 0.15: The Third Generation (GK Series) is related to the Honda Shuttle.
  Fact 0.16: The Third Generation (GK Series) is related to the Acura CDX.
  Fact 0.17: The Third Generation (GK Series) has a 1.2 L L12B i-VTEC I4 petrol engine.
  Fact 0.18: The Third Generation (GK Series) has a 1.3 L L13B Earth Dreams i-VTEC I4 engine.
  Fact 0.19: The Third Generation (GK Series) has a 1.5 L L15A I4 petrol engine.
  Fact 0.20: The Third Generation (GK Series) has a 1.5 L L15B1 I4 petrol engine.
  Fact 0.21: The Third Generation (GK Series) has a 1.5 L L15B Earth Dreams i-VTEC I4 engine.
  Fact 0.22: The Third Generation (GK Series) has a 1.5 L i-DTEC diesel engine.
  Fact 0.23: The 5-speed manual transmission is available in China only.
  Fact 0.24: The Third Generation (GK Series) has a 6-speed manual transmission.
  Fact 0.25: The Third Generation (GK Series) has a 7-speed semi-automatic (DCT) transmission.
  Fact 0.26: The Third Generation (GK Series) has a CVT automatic transmission.
  Fact 0.27: The Third Generation (GK Series) has a wheelbase of 2,530 mm (99.6 in).
  Fact 0.28: The Third Generation (GK Series) has a length of 3,955 mm (155.7 in).
  Fact 0.29: The Third Generation (GK Series) has a width of 1,695 mm (66.7 in).
  Fact 0.30: The Third Generation (GK Series) has a height of 1,525 to 1,550 mm (60.0 to 61.0 in).
  Fact 0.31: The Third Generation (GK Series) has a curb weight of 1,020 to 1,080 kg (2,249 to 2,381 lb).
  Fact 0.32: The hybrid version of the Third Generation (GK Series) has a curb weight of 1,080 to 1,170 kg (2,381 to 2,579 lb).

Retrieved items:
<item 1>
Honda Fit
The base model receives body - color exterior mirrors ( previously black ) , standard security system and new wheel covers for 2012 . Canadian trims include the
DX ( two speakers , black door handles and side mirrors ) , DX - A ( adds factory - installed air conditioning ) , LX ( adds among other things , body - colored
door handles and side mirrors , 15 - inch lightweight alloy wheels from the 2006 -- 11 Civic Hybrid and DX - G , rear spoiler , power locks and remote entry ,
two speakers for a total of four , cruise control ) , and Fit Sport ( adds USB jack and Vehicle Stability Assist ) . The Fit Sport is similar to its American
counterpart , however a navigation system is not offered as an option . There are plans to build subcompact cars , which many rumoured to be Fit , in Honda 's
new plant in Mexico . NHTSA Crash Test Ratings : Frontal Driver and Passenger : Side Driver Impact : Side Rear Passenger Impact : Rollover : Brazil The Sumare -
built new Fit was introduced in October 2008 in Brazil at the Sao Paulo Auto Show .
</item 1>

<item 2>
Honda Fit
With the fuel tank placed in the center plus an ultra-thin electric motor and compact IPU , it claims to achieve an efficient weight distribution . In an
interview , Kohei Hitomi , the project leader of Fit hybrid , admitted that lowering CO further will affect some attributes that buyers of the car value : a
bigger electric motor and battery sacrifice spaces utility , and improving aerodynamics might make side mirrors smaller . He said there are benefits for
customers to keep all the versatility . The drivetrain is the same drive system as the Honda Insight . The advantages of the system are low cost , low weight
and that it can be combined with either manual transmission or continuous variable transmission . The car is better suited than its diesel competitors in areas
where superminis are most common : city centres . It is more economical in urban conditions . Thanks to the instant torque of the electric motor , it has a much
better engine response . The hybrid version was not available in the U.S. According to Koichi Kondo , Executive Vice President of Honda : `` As for the future ,
it 's open to question . We will carefully be watching the market situation . ''
</item 2>

<item 3>
Honda Fit
In addition , the Fit Sport now comes with carpeted floor mats . As of the 2011 model year , the base Fit comes with a remote entry , 15 - inch wheels with
covers , power windows , power locks , cruise control , a tilt and telescoping steering wheel , a 160 - watt AM / FM / CD audio system with four speakers , a
USB audio interface , a 12 - volt power outlet , and other features . The Fit Sport adds a ground appearance package , 16 - inch ( 410 mm ) alloy wheels ,
paddle shifters , a navigation system , a security system , fog lights , chrome exhaust finisher , a leather - wrapped steering wheel , steering wheel - mounted
audio controls , floor mats , a 6 - speaker AM / FM / CD audio system , and other features . 2010 Honda Fit Sport 2012 Honda Fit Sport ( US ) For 2012 , the Fit
features revised floor , front fenders , A-pillars , and front corner - window glass . The Fit Sport features a revised grille , dark `` machined surface ''
finish alloy wheels , a black monochrome theme interior with chrome accents .
</item 3>

<item 4>
Honda Fit
525 -- 1,550 mm ( 60.0 -- 61.0 in ) Curb weight 1,020 -- 1,080 kg ( 2,249 -- 2,381 lb ) 1,080 -- 1,170 kg ( 2,381 -- 2,579 lb ) ( Hybrid ) The third generation
Fit retains the overall design concept of previous generations , notably the center - located fuel tank and multi-configuration interior concept , marketed as
Magic Seating . Honda 's all - new `` Global Small Car Platform '' employing ultra-high strength 780 MPa yield steel for 27 % of its bodywork -- and a shorter
overall length ( - 1.6 inches ) , longer wheelbase ( + 1.2 inch ) increased rear legroom ( + 4.8 inches ) and increased passenger volume ( + 4.9 cu ft )
compared to its previous generation . Body panels are both welded and bolted to the frame in a hybrid monocoque and spaceframe fusion -- and rear torsion beam
suspension is more compact , no longer using an anti-sway bar to maximize interior and cargo space . The HR - V shares its platform with the Fit .
</item 4>

<item 5>
Honda Fit
Indonesia Santa Rosa , Philippines Guangzhou , China ( Guangqi Honda ) Sumaré , Brazil ( Honda Brazil ) Designer Mitsuhiro Honda ( 1999 ) Body and chassis
Related Honda Airwave Honda City / Fit Aria Powertrain Engine 1.2 L L12A I4 ( petrol ) 1.3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed
manual 5 - speed automatic CVT automatic Dimensions Wheelbase 2,450 mm ( 96.5 in ) Length 3,845 mm ( 151.4 in ) ( Worldwide ) 4,000 mm ( 157.4 in ) ( U.S. &
Canada ) Width 1,675 mm ( 65.9 in ) Height 1,525 mm ( 60.0 in ) Curb weight 1,084 kg ( 2,390 lb ) The Honda Fit debuted in June 2001 in Japan and immediately
became a big hit . At its introduction in 2001 , it won the Car of the Year Japan Award .
</item 5>

<item 6>
Honda Fit
Malaysia Guangzhou , China ( Guangqi Honda ) Celaya , Guanajuato , Mexico ( from Feb 2014 ) Tapukara , India ( from Jan 2015 ) Designer Riku Wada ( 2010 ) Body
and chassis Related Honda City Honda HR - V Honda Freed Honda Shuttle Acura CDX Powertrain Engine 1.2 L L12B i - VTEC I4 petrol 1.3 L L13B Earth Dreams i - VTEC
I4 1.5 L L15A I4 ( petrol ) 1.5 L L15B1 I4 ( petrol ) 1.5 L L15B Earth Dreams i - VTEC I4 1.5 L i - DTEC diesel Transmission 5 - speed manual ( China Only ) 6 -
speed manual 7 - speed semi-automatic ( DCT ) CVT automatic Dimensions Wheelbase 2,530 mm ( 99.6 in ) Length 3,955 mm ( 155.7 in ) Width 1,695 mm ( 66.7 in )
Height 1,525 -- 1,550 mm ( 60.0 -- 61.
</item 6>

<item 7>
Honda Fit
All models feature ECON Mode for enhancing real - world fuel economy . Hybrid model has Eco Assist ( Ecological Drive Assist System ) to help the driver engage
in fuel - efficient driving practices . It went on sale on June 16 , 2011 , in Japan . The Fit Shuttle comes with a double - hinged two - piece load floor that
both allows access to an under - floor storage bin and also functions as a divider for the cargo area . Through better sound dampening and insulation , the Fit
Shuttle is pledged to be as quiet as a mid-size sedan . Japan JDM - spec interior Japanese sales began in October 2007 . Facelifted models , and the hybrid
version , arrived in October 2010 . Models available in Japan are the 1.3 L and 1.5 L variants consisting of the 13G , 15XH and the top - of - the - line RS .
In 2007 , aftermarket tuner Mugen , introduced a new line of accessories for the Japanese market 2008 Honda Fit . On 17 January 2008 , they introduced the Mugen
Fit F154SC at the 2008 Tokyo Auto Salon .
</item 7>

<item 8>
Honda Fit
Gawker Media . Retrieved 2008 - 03 - 19 . Jump up ^ Bell , Damon . `` The Consumer Guide to 2009 Subcompact Cars '' . Consumer Guide Automotive . HowStuffWorks
. Archived from the original on 2009 - 08 - 30 . Retrieved 2009 - 08 - 25 . The most expensive subcompact , the top - trim - level Honda Fit with an optional
navigation system , has a still - modest base price of $18,760 . Jump up ^ `` Honda Fit production could shift to Ohio '' . The Columbus Dispatch . Archived
from the original on 2013 - 01 - 21 . Retrieved 2009 - 10 - 22 . Jump up ^ `` 2011 Honda Fit Receives More Standard Features ; VSA Added to All Models '' .
theautochannel.com . Jump up ^ `` 2009 Honda Fit Review '' . JB car pages . Retrieved 2009 - 03 - 09 . Jump up ^ `` 2012 Honda Fit Receives Updates and More
Features '' ( Press release ) .
</item 8>

<item 9>
Honda Fit
3 L L13A I4 ( petrol ) 1.5 L L15A I4 ( petrol ) Transmission 5 - speed manual 6 - speed manual 5 - speed automatic CVT automatic ( Japan ) Dimensions Wheelbase
2,500 mm ( 98.4 in ) Length 3,985 mm ( 156.9 in ) ( 2009 -- 11 ) 4,100 mm ( 161.4 in ) ( 2012 -- 13 ) 4,410 mm ( 173.6 in ) ( Fit Shuttle ) Width 1,695 mm (
66.7 in ) Height 1,525 mm ( 60.0 in ) 1,540 mm ( 60.6 in ) ( Fit Shuttle ) Curb weight 1,070 kg ( 2,359 lb ) The second generation Fit debuted on 17 October
2007 at the Tokyo Motor Show . At its introduction in 2007 , it won the Car of the Year Japan Award for the second time . The vehicle offered a longer wheelbase
than its predecessor and is wider and longer overall . Overall height is unchanged , while interior height increased by . 3 in .
</item 9>

<item 10>
Honda Fit
National Highway Traffic Safety Administration ( NHTSA ) crash test ratings . The engine is a 1.5 L ( 92 cu in ) i - VTEC 4 - cylinder with 117 horsepower at
6600 rpm mated to a five - speed manual transmission or a five - speed automatic transmission . The second generation Honda Fit went on sale in North America
around August 26 , 2008 , for the 2009 model year . It was considered the most expensive within the subcompact class when equipped as such , but was also
described as modestly priced . It was reported in September 2009 that Honda is considering producing the Fit in North America in response to the increasing
strength of the Japanese yen against the U.S. dollar . For the 2011 models , Vehicle Stability Assist , cruise control , remote entry and USB audio interface
became standard across all trim levels . In addition , the Fit Sport now comes with carpeted floor mats .
</item 10>
=== END ITEM id=602cd316b268d682 ===
