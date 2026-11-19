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

=== ITEM id=48dccc4e3f46c3b6 ===
Gold sentences and facts:
Sentence 0: We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving.
  Fact 0.0: People identify with the superhero's struggles.
  Fact 0.1: People identify with the superhero's little glimmers of connection and triumph.
  Fact 0.2: Identifying with the superhero makes the superhero the perfect superhero for Thanksgiving.
Sentence 1: Apart from the food - arguably the main component of the day - the holiday is a beloved time of year when Americans come together to celebrate what they are thankful for, either with family or friends.
  Fact 1.0: Thanksgiving is a holiday.
  Fact 1.1: The food is arguably the main component of Thanksgiving day.
  Fact 1.2: Thanksgiving is a beloved time of year.
  Fact 1.3: Americans come together during Thanksgiving to celebrate what they are thankful for.
  Fact 1.4: Americans celebrate what they are thankful for with family or with friends during Thanksgiving.

Retrieved items:
<item 1>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Aunt May has just gotten out of the hospital, and even noted jerk “Flash” Thompson has given Peter hell for how awful he’s been acting.

Peter, attempting to make things right, opts to cook the Thanksgiving meal all by himself, but mostly succeeds in ruining the kitchen. No worries — Gwen Stacy
and her father, along with a recuperating Aunt May and her doctor, all help out, and the episode concludes with a pleasant dinner. There are no big turns or
twists, aside from Aunt May revealing that she’s publishing a cookbook. It’s just simple, earned solace in a life marked by chaos. Peter even gets a kiss from
Gwen on his porch, a sequence conducted with a John Hughes sense of satisfying romantic flair.

Various Spider-Man comics have also dabbled in seeing what Thanksgiving looks like when you’re young and arachnid-themed, but real life has associated
Spider-Man with the holiday, too. Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade.

Graph facts:
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Harry
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Mary Jane
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Thanksgiving
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> capitalism
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Spider-Man

Entities: Aunt May, Thanksgiving is Spider-Man’s holiday, Peter, Spider-Man, Flash Thompson, Gwen Stacy, John Hughes, Macy’s, Macy’s Thanksgiving Parade,
Polygon, Marvel, Thanksgiving
</item 1>

<item 2>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Sure, everyone in that room is trying to hold things together. Mary Jane wants to impress Norman, Harry wants to impress Norman and Mary Jane, Peter loves Mary
Jane but doesn’t dare hurt Harry, and Aunt May is finishing what’s presumably a dope turkey, in an attempt to care for all of them.

It’s the trying that counts: Life can be hard, weird, and cruel, but while sitting down for a meal with our nearest and dearest, maybe for at least a little
while, we won’t need to have it all figured out.

Not all Spider-Man Thanksgiving dinners end up falling apart, and some succeed in reminding Peter that he isn’t alone. The first season of The Spectacular
Spider-Man ends with a bang: One of Peter’s closest pals, Eddie Brock, has become Venom, and has threatened everyone Peter holds dear. This comes just after
Peter himself was infected by the symbiote, and went through the now-iconic throes of pushing everyone away.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Mary Jane
- Mary Jane -WORKS_FOR-> The Daily Bugle
- Mary Jane -PARTICIPATED_IN-> Wake Up quest
- Mary Jane -PARTICIPATED_IN-> Wake Up
- Mary Jane -PARTICIPATED_IN-> Good Men

Entities: Mary Jane, Thanksgiving, Harry, Spider-Man, Aunt May, Thanksgiving is Spider-Man’s holiday, Norman, Peter, The Spectacular Spider-Man, Eddie Brock,
Venom, symbiote, Polygon
</item 2>

<item 3>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
perform in the parade Reuters Macy's Thanksgiving Day Parade takes place in New York The Charlie Brown balloon is inflated Getty Macy's Thanksgiving Day Parade
takes place in New York The Tom Turkey float is in place to start the parade Reuters

Later, US Presidents including George Washington, John Adams and James Madison called for days of thanks throughout their presidencies.

However, it was not until 1863, during the Civil War, that President Abraham Lincoln declared a national Thanksgiving Day would be held each November to be
celebrated by all of the country.

When is Thanksgiving?

Thanksgiving takes place each year on the fourth Thursday of November.

This year, Thanksgiving falls on 23 November.

Why do Americans celebrate it and what do they do?

Although Thanksgiving may originally have had religious significance, the day has become a mostly secular holiday.

Most Americans consider the holiday a day to gather and express their thanks through food, family, and football – with multiple NFL teams playing on the
holiday.

During some Thanksgiving celebrations, people write down what they are thankful for and then read aloud from the pieces of paper.

Graph facts:
- Brock Purdy -MENTIONS-> NFL
- Raiders vs. Lions live score, updates, highlights from NFL 'Monday Night Football' game -MENTIONS-> NFL
- Fans spot Travis Kelce wearing Taylor Swift-themed friendship bracelet before she attended his game -MENTIONS-> NFL
- Chiefs vs. Packers live score, updates, highlights from NFL 'Sunday Night Football' game -MENTIONS-> NFL
- Christmas Day preview: 49ers, Ravens square off in potential Super Bowl sneak peek -MENTIONS-> NFL

Entities: NFL, Thanksgiving, New York, Civil War, Why do Americans celebrate Thanksgiving?, Macy's Thanksgiving Day Parade, George Washington, John Adams, James
Madison, Abraham Lincoln, The Charlie Brown balloon, The Tom Turkey float, The Independent
</item 3>

<item 4>
Graph query row:
article: Thanksgiving is Spider-Man’s holiday
connected_entity: Thanksgiving
entity_types: ['__KGBuilder__', '__Entity__', 'Event']
</item 4>

<item 5>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feeding friends and family is perhaps one of the most humane acts we can pursue. It’s less about the spirit of giving, and more about admitting that people
have innate, basic needs, like food and social comfort, and that those needs are best fulfilled when people work in tandem. Honestly, we should try to do it
more on every other day of the year: The number of people volunteering to help feed the homeless and families in need peaks around November and December, but
that energy needs to be carried through the preceding 10 months, too.

Spider-Man represents these needs, even though very few of us have to balance photojournalist work, a fraught dating life, and pummeling Dr. Octopus. Peter
Parker is often the most financially strapped among his Avengers buddies, and frequently the loneliest, too. Those are recognizable traits among many young
people, even those without radioactive-spider powers. When you’re growing up and trying to figure out the world, it’s easy to feel lost and isolated.

Graph facts:
- Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit -MENTIONS-> Thanksgiving
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Thanksgiving
- FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Thanksgiving
- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology -MENTIONS-> Thanksgiving
- Why do Americans celebrate Thanksgiving? -MENTIONS-> Thanksgiving

Entities: Thanksgiving, Spider-Man, Dr. Octopus, Avengers, Thanksgiving is Spider-Man’s holiday, Polygon
</item 5>

<item 6>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Marvel built its comic book revolution on the back of one idea: rendering colorful superheroes as relatable. In the mid 20th century, DC do-gooders were
essentially square-jawed Sunday school teachers, which let Marvel corner the market on heroes with human depth and fragility. The Fantastic Four were vulnerable
to insecurity and in-fighting. The X-Men represented the cost of bigotry on a wide scale.

But Spider-Man represented this new wave best. Consumed by youthful everyman angst and desperate to find balance in his life, Spider-Man is broadly sympathetic.
We identify with his struggles and his little glimmers of connection and triumph — which makes him the perfect superhero for Thanksgiving. And Spider-Man
writers know it.

Thanksgiving has a complicated history, tied to the roots of American colonialism, and more recently processed through a lens of joyful capitalism. But feeding
friends and family is perhaps one of the most humane acts we can pursue.

Graph facts:
- Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit -MENTIONS-> Thanksgiving
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Thanksgiving
- FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Thanksgiving
- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology -MENTIONS-> Thanksgiving
- Why do Americans celebrate Thanksgiving? -MENTIONS-> Thanksgiving

Entities: Thanksgiving, capitalism, Spider-Man, The Fantastic Four, The X-Men, American colonialism, Thanksgiving is Spider-Man’s holiday, Marvel, DC, Polygon
</item 6>

<item 7>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
Spider-Man is the only Marvel character to rate a balloon in the Macy’s Thanksgiving Parade. Other Marvel characters have appeared on floats, but only
Spider-Man has been inflated to 80 feet and pulled through the Upper West Side. (That also makes him the only Marvel character to have his head horrifically
torn open by a tree branch.)

From his first comic book appearance, Spider-Man has been a reminder that life is hard and complicated, and being a superhero doesn’t preclude anyone from
experiencing ordinary frustrations, setbacks, and confusion. But no matter how Spider-Man’s Thanksgiving escapades turn out, they remind readers and viewers
that the holiday is about the hope of mutual connection, support, and nurturing. Even if the girl of your dreams is out of reach and the Green Goblin is on your
case, a table, some friends, and Aunt May’s rad turkey might just make everything better for a little while.

Graph facts:
- Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit -MENTIONS-> Thanksgiving
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Thanksgiving
- FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Thanksgiving
- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology -MENTIONS-> Thanksgiving
- Why do Americans celebrate Thanksgiving? -MENTIONS-> Thanksgiving

Entities: Thanksgiving, Thanksgiving is Spider-Man’s holiday, Spider-Man, Marvel, Macy’s Thanksgiving Parade, Upper West Side, Aunt May, Green Goblin, Polygon
</item 7>

<item 8>
Why do Americans celebrate Thanksgiving? (The Independent - Life and Style, 2023-11-11T09:04:08+00:00)
What is Thanksgiving and where did it come from?

Thanksgiving is a national holiday in the United States celebrated every fourth Thursday in November.

The annual feast is in honour of the “first” Thanksgivings in America, in 1619 in Virginia, and in 1621, when colonists in Plymouth, Massachusetts, later known
as the Pilgrims, allegedly shared a meal with the Wampanoag Indians, who were native to the land.

The later feast was in honour of the help the Pilgrims received from the Native Americans in cultivating crops and surviving their first harsh winter and it
lasted three days.

Macy's Thanksgiving Day Parade takes place in New York Show all 15 1 / 15 Macy's Thanksgiving Day Parade takes place in New York Macy's Thanksgiving Day Parade
takes place in New York Performers walk in front of Macy's Tom Turkey float in the Thanksgiving Day Parade in New York AP Macy's Thanksgiving Day Parade takes
place in New York The Ronald McDonald balloon is inflated Getty Macy's Thanksgiving Day Parade takes place in New York Sinclair's DINO balloon flies as the

Graph facts:
- After losing her leg to the flu, Virginia woman urges people to get vaccinated: ‘Don’t waste time’ -MENTIONS-> Virginia
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> Virginia
- The obsessive tormenter who made professors’ lives miserable -MENTIONS-> Virginia
- Best sportsbook bonus offers for NFL Monday Night Football Eagles vs. Seahawks: Claim over $5,000 in bonuses from Bet365, BetMGM, BetRivers, Caesars
Sportsbook, DraftKings and FanDuel -MENTIONS-> Virginia
- Sam Altman backs teens’ startup, Google unveils the Pixel 8 and TikTok tests an ad-free tier -MENTIONS-> Virginia

Entities: Virginia, Pilgrims, Thanksgiving, Native Americans, New York, United States, Why do Americans celebrate Thanksgiving?, First Thanksgiving in 1619,
First Thanksgiving in 1621, Plymouth, Massachusetts, Wampanoag Indians, Macy's Thanksgiving Day Parade, Macy's, Tom Turkey float, Ronald McDonald balloon
</item 8>

<item 9>
Thanksgiving is Spider-Man’s holiday (Polygon, 2023-11-23T16:04:00+00:00)
But feelings of loneliness spike during the holidays, meaning that there’s a good chance you’ll feel even more like Peter Parker when late November rolls
around.

It’s what makes the Thanksgiving scene in the 2002 Spider-Man film so engaging. That whole movie is an exercise in heart-on-your-sleeve sweetness. Everyone
who’s seen it knows the Thanksgiving dinner rapidly descends into chaos, with Norman Osborn ominously sticking his fingers into Aunt May’s sweet-potato
casserole, figuring out Spider-Man’s secret identity, and delivering a grossly sexist diatribe to his son Harry, leaving Harry and Mary Jane at odds with each
other. Aunt May and Peter clearly wind up with a ton of leftovers after everyone else storms out.

But for Peter, who’s just lost his Uncle Ben and has been facing the initial trials of being Spider-Man, there’s a nice moment of personal respite at the
opening of that scene, where he walks into a room where he knows he’s loved, bearing an offering of cranberry sauce,.

Graph facts:
- Mary Jane in Marvel’s Spider-Man 2 Is a Disguised Superhero, in a Good Way -MENTIONS-> Mary Jane
- Mary Jane -WORKS_FOR-> The Daily Bugle
- Mary Jane -PARTICIPATED_IN-> Wake Up quest
- Mary Jane -PARTICIPATED_IN-> Wake Up
- Mary Jane -PARTICIPATED_IN-> Good Men

Entities: Mary Jane, Thanksgiving, Harry, Spider-Man, Norman Osborn, Aunt May, Uncle Ben, 2002 Spider-Man film, Thanksgiving dinner, Thanksgiving is
Spider-Man’s holiday, Polygon
</item 9>

<item 10>
From a Turkey trot to a movie marathon, Friendsgiving celebrations that aren’t a sit-down dinner (The Independent - Life and Style, 2023-11-17T16:20:48+00:00)
While it’s often viewed as a day for Americans to express what they’re grateful for, that doesn’t mean that everyone will celebrate. More specifically, the
event has been viewed as deeply controversial over the years, due to the contentious history of the first Thanksgiving celebration in the 17th century.

The holiday has been linked to instances of cultural appropriation, with some Americans believing that its history is frequently white-washed, prompting them to
condemn or ignore Thanksgiving as a whole.

However, the holiday itself has inspired another tradition of Friendsgiving – which can be held before, during, or even after the 23rd of November this year.
The activity typically consists of a celebration with close pals since some Americans don’t have the opportunity to return home for the holidays.

Over the years, people have celebrated Friendsgiving in a similar vein as Thanksgiving, with everyone sitting around the table for one big feast. It’s also
often done in a potluck style, with each guest bringing their own dish to contribute.

Graph facts:
- Cranberries: A Thanksgiving Staple With a Rich History and Unique Biology -MENTIONS-> Thanksgiving
- Fantasy Football RB Rankings Week 14: Who to start, best sleepers at running back -MENTIONS-> Thanksgiving
- FanDuel Picks Week 14: Best NFL DFS lineup advice for daily fantasy football tournaments -MENTIONS-> Thanksgiving
- Thanksgiving is Spider-Man’s holiday -MENTIONS-> Thanksgiving
- Meet the American who made cranberries a Turkey Day tradition, Marcus Urann, farmer with can-do spirit -MENTIONS-> Thanksgiving

Entities: Thanksgiving, cultural appropriation, Friendsgiving, first Thanksgiving celebration, white-washing, From a Turkey trot to a movie marathon,
Friendsgiving celebrations that aren’t a sit-down dinner, The Independent
</item 10>
=== END ITEM id=48dccc4e3f46c3b6 ===

=== ITEM id=e2e17ed50c5bebd4 ===
Gold sentences and facts:
Sentence 0: “The company’s approach also raises concerns regarding the GDPR,” Pachl further noted.
  Fact 0.0: The company's approach raises concerns regarding the GDPR.
  Fact 0.1: Pachl further noted this.
Sentence 1: Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents.
  Fact 1.0: Meta wants to shift back to the app stores the burden of monitoring teens' social media usage.
  Fact 1.1: Meta wants to shift back to parents the burden of monitoring teens' social media usage.
Sentence 2: Al Saadi said that her frustration with Meta is common among Palestinians, both in occupied territories and across the diaspora.
  Fact 2.0: Al Saadi said that her frustration with Meta is common among Palestinians.
  Fact 2.1: Al Saadi said that her frustration with Meta is common among Palestinians in occupied territories.
  Fact 2.2: Al Saadi said that her frustration with Meta is common among Palestinians across the diaspora.

Retrieved items:
<item 1>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
At the recommendation of its Oversight Board, Meta conducted a third-party due diligence report about the platform’s moderation during the May 2021 conflict.
The report found that Arabic content was flagged as potentially violating at significantly higher rates than Hebrew content was, and was more likely to be
erroneously removed. The report noted that Meta’s moderation system may not be as precise for Arabic content as it was for Hebrew content, because the latter is
a “more standardized language,” and suggested that reviewers may lack the linguistic and cultural competence to understand less common Arabic dialects like
Palestinian Arabic.

Has anything improved?

Meta committed to implementing policy changes based on the report’s recommendations, such as updating its keywords associated with dangerous organizations,
disclosing government requests to remove content and launching a hostile speech classifier for Hebrew content. Abukhater added that Meta has improved its
response to harassment, at least in comparison to other social media platforms like X (formerly Twitter).

Graph facts:
- Why Mozilla is betting on a decentralized social networking future -MENTIONS-> moderation
- Elon Musk’s X faces first DSA probe in EU over illegal content risks, moderation, transparency and deceptive UX design -MENTIONS-> moderation
- Alexis Ohanian says he’s a ‘techno optimist,’ despite social media’s impact on society -MENTIONS-> moderation
- CEO David Baszucki’s mission to make Roblox a billion-player platform -MENTIONS-> moderation
- Pebble, the Twitter alternative previously known as T2, is shutting down -MENTIONS-> moderation

Entities: moderation, Arabic content, Abukhater, Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices, Meta, Oversight
Board, third-party due diligence report, May 2021 conflict, Hebrew content, Palestinian Arabic, hostile speech classifier, government requests to remove
content, TechCrunch, harassment, X
</item 1>

<item 2>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
A newly unredacted version of the multi-state lawsuit against Meta alleges a troubling pattern of deception and minimization in how the company handles kids
under 13 on its platforms. Internal documents appear to show that the company’s approach to this ostensibly forbidden demographic is far more laissez-faire than
it has publicly claimed.

The lawsuit, filed last month, alleges a wide spread of damaging practices at the company relating to the health and well-being of younger people using it. From
body image to bullying, privacy invasion to engagement maximization, all the purported evils of social media are laid at Meta’s door — perhaps rightly, but it
also gives the appearance of a lack of focus.

In one respect at least, however, the documentation obtained by the attorneys general of 42 states is quite specific, “and it is damning,” as AG Rob Bonta of
California put it. That is in paragraphs 642 through 835, which mostly document violations of the Children’s Online Privacy Protection Act, or COPPA.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, California, multi-state lawsuit against Meta, Rob Bonta, Children’s Online Privacy Protection Act, Meta turned a blind eye to kids
on its platforms for years, unredacted lawsuit alleges
</item 2>

<item 3>
Graph query row:
article: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices
page: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
passage: Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices Instagram users suspect Meta of shadowbanning content about
Palestine. The platform has a history of inherent bias.

Earlier this year, Palestinian-American filmmaker Khitam Jabr posted a handful of Reels about her family’s trip to the West Bank. In the short travel vlogs,
Jabr shared snippets of Palestinian culture, from eating decadent meals to dancing at her niece’s wedding.

“I hadn’t been in a decade, so it’s just like, life abroad,” Jabr told TechCrunch.

“I would get [anti-Palestine] comments,” she recalled. “And I couldn’t respond [to them] or use my account for 24 hours. I wasn’t even posting anything about
the occupation. But fast forward to now and the same shit’s happening.”

In the aftermath of Hamas’ attack on Israelis, Israel’s retaliatory airstrikes and total blockade — cutting access to electricity, water and vital supplies —
have devastated Gaza.
</item 3>

<item 4>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“It feeds into the bias against Palestinian narrative genuinely. It really takes the balance against Palestinians in a situation where there’s a huge asymmetry
of power.”

A history of suppression

Content about Palestine is disproportionately scrutinized, as demonstrated during the last severe violent outbreak between Hamas and Israel two years ago. Amid
the violence following the May 2021 court ruling to evict Palestinian families from Sheikh Jarrah, a neighborhood in occupied East Jerusalem, users across
Facebook and Instagram accused Meta of taking down posts and suspending accounts that voiced support for Palestinians.

The digital rights nonprofit Electronic Frontier Foundation (EFF) described Meta’s actions in 2021 as “systemic censorship of Palestinian voices.” In its 2022
report of Palestinian digital rights, 7amleh said that Meta is “still the most restricting company” compared to other social media giants in the extent of its
moderation of the Palestinian digital space.

Graph facts:
- How Chinese firm linked to repression of Uyghurs aids Israeli surveillance in West Bank -MENTIONS-> occupied East Jerusalem
- Hitting the Books: We are the frogs in the boiling pot, it's time we started governing like it. -MENTIONS-> Electronic Frontier Foundation
- Jalal Abukhater -WORKS_FOR-> 7amleh
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> Sheikh Jarrah
- John Perry Barlow -FOUNDED-> Electronic Frontier Foundation

Entities: Palestinian voices, 7amleh, moderation bias, Severe violent outbreak between Hamas and Israel, Sheikh Jarrah, occupied East Jerusalem, Electronic
Frontier Foundation, Palestinian digital rights, Systemic censorship of Palestinian voices, Moderation of the Palestinian digital space, Meta has a moderation
bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices, Meta, TechCrunch, Court ruling to evict Palestinian families from Sheikh Jarrah, Hamas
</item 4>

<item 5>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
The former employee had provided to news outlets a treasure trove of documents that appeared to indicate Meta understood the harms it was causing — including
things like body image issues among teens — but didn’t take action. Instagram head Adam Mosseri was later hauled before Congress in December 2021 to defend the
app’s teen safety record, but so far, legislators have not come to any decision about how to regulate teen usage of these platforms.

Instead, Meta began regulating itself, adding new defaults and features that would restrict teens’ access to content on its apps, protect their privacy and
limit ad targeting to teens, as well as introducing parental controls.

In more recent weeks, a second Meta whistleblower, Arturo Bejar, has come forward to express concerns that Instagram’s approach to protecting teens wasn’t
working to keep them safe from sexual predators and unwanted sexual advances and harassment. Hired as a consultant on the matter, Bejar eventually blew the
whistle on Meta after seeing issues remain unresolved following years of work.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Arturo Bejar, body image issues among teens, sexual predators, unwanted sexual advances and harassment, Adam Mosseri hauled before
Congress, Meta seeks legislation that would require parents to approve teens’ app downloads, Congress, Instagram, teen safety, Adam Mosseri
</item 5>

<item 6>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
During the last violent outbreak between Hamas and Israel in 2021, a report commissioned by Meta and conducted by a third party concluded that the company’s
actions had an “adverse human rights impact” on Palestinian users’ right to freedom of expression and political participation.

Meta communications director Andy Stone declined to comment on the record, and pointed TechCrunch to Meta’s newsroom post about the conflict, which was updated
on Oct. 18 with a statement addressing the shadowbanning claims.

“Our policies are designed to keep people safe on our apps while giving everyone a voice,” the post stated. “We apply these policies equally around the world
and there is no truth to the suggestion that we are deliberately suppressing voice … We can make errors and that is why there is an appeals process for people
to tell us when they think we have made the wrong decision, so we can look into it.”

The belief that Meta shadowbans, or limits the visibility of, content about Palestine is not new.

Graph facts:
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> moderation
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> EU
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> shadowbanning
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Stone
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Threads

Entities: shadowbanning, Andy Stone, violent outbreak between Hamas and Israel, deliberately suppressing voice, Meta has a moderation bias problem, not just a
‘bug,’ that’s suppressing Palestinian voices, Meta, TechCrunch, Hamas, Israel, Palestine
</item 6>

<item 7>
Meta seeks legislation that would require parents to approve teens’ app downloads (TechCrunch, 2023-11-15T17:13:10+00:00)
Meta wants to shift the burden of monitoring social media usage among teens back to the app stores — and to parents. In a blog post published today, Meta’s
global head of Safety, Antigone Davis, argues that parents should be responsible for approving their teens’ app downloads and staked Meta’s position in
supporting federal legislation that would require parental approval for app downloads for users under the age of 16.

Guiding its reasoning, Meta pointed to recent Pew research that indicated that 81% of U.S. adults were in favor of requiring parental consent for teens to
create social media accounts. While this may speak to popular opinion around parental sentiment, it doesn’t mean that parents have thought through who should be
responsible for preventing or allowing teens’ access to social media, or how those apps should operate to protect teens.

A coalition of 42 states and D.C. are currently suing Meta over its harms to teens and young users, in part referencing findings from Meta whistleblower Frances
Haugen.

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Antigone Davis, coalition of 42 states and D.C., lawsuit against Meta over harms to teens and young users, Frances Haugen, harms to
teens and young users, parental approval for teens’ app downloads, parental responsibility for approving teens’ app downloads, Meta seeks legislation that would
require parents to approve teens’ app downloads, Pew Research
</item 7>

<item 8>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
“And so I think that is what is the most problematic right now about that particular rule, and the way that it’s enforced.”

Over the 13 days leading up to the ceasefire between Hamas and Israel, 7amleh documented more than 500 reports of Palestinian “digital rights violations,”
including the removal and restriction of content, hashtags and accounts related to the conflict.

Meta blamed some of the instances of perceived censorship to technical issues, like one that prevented users in Palestine and Colombia from posting Instagram
Stories. It attributed others to human error, like blocking the hashtag for Al-Aqsa Mosque, the holy site where Israeli police clashed with Ramadan worshippers,
because it was mistaken for a terrorist organization. The company also blocked journalists in Gaza from WhatsApp without explanation.

The same month, a group of Facebook employees filed internal complaints accusing the company of bias against Arab and Muslim users. In internal posts obtained
by BuzzFeed News, an employee attributed the bias to “years and years of implementing policies that just don’t scale globally.”

Graph facts:
- Jalal Abukhater -WORKS_FOR-> 7amleh
- People are turning to Snap Map for firsthand perspectives from Gaza -MENTIONS-> censorship
- Leen Al Saadi -SAID-> censorship
- The people who ruined the internet -MENTIONS-> censorship
- Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices -MENTIONS-> Stone

Entities: censorship, Palestinian digital rights violations, 7amleh, moderation bias, Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing
Palestinian voices, Meta, ceasefire between Hamas and Israel, Colombia, technical issues, Al-Aqsa Mosque, Israeli police, clash between Israeli police and
Ramadan worshippers, Ramadan worshippers, WhatsApp, Facebook employees
</item 8>

<item 9>
Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices (TechCrunch, 2023-10-19T23:43:15+00:00)
Although harassment and abuse are still rampant on Instagram and Facebook, he said, the company has been responsive to suspending accounts with patterns of
targeting other users.

The company has also made more contact with regional Palestinian organizations since 2021, York added, but it’s been slow to implement recommendations from EFF
and other advocacy groups. It’s “very clear” that Meta is not putting the same resources behind Arabic and other non-English languages, York said, compared to
the attention Meta gives to countries that have the most regulatory pressure. Moderation of English and other European languages tends to be more comprehensive,
for example, because the EU enforces the Digital Services Act.

In Meta’s response to the report, Miranda Sissons, the company’s director of human rights, said that Meta was “assessing the feasibility” of reviewing Arabic
content by dialect. Sissons said that the company has “large and diverse teams” who understand “local cultural context across the region,” including in
Palestine.

Graph facts:
- EU fires urgent warning at Elon Musk’s X over illegal content and disinformation following Hamas attacks -MENTIONS-> EU
- Amazon’s iRobot purchase sucks up formal competition concerns in EU -MENTIONS-> EU
- ChatGPT: Everything you need to know about the AI-powered chatbot -MENTIONS-> EU
- Biden’s AI EO hailed as broad, but not deep without legislation to match -MENTIONS-> EU
- After warning, the EU opens investigation into X over alleged disinformation -MENTIONS-> EU

Entities: EU, Arabic content, moderation bias, EFF, Meta has a moderation bias problem, not just a ‘bug,’ that’s suppressing Palestinian voices, Meta, York,
Miranda Sissons, regional Palestinian organizations, other advocacy groups, non-English languages, TechCrunch, human rights, Instagram, Digital Services Act
</item 9>

<item 10>
Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges (TechCrunch, 2023-11-27T21:20:32+00:00)
This law created very specific restrictions around young folks online, limiting data collection and requiring things like parental consent for various actions,
but a lot of tech companies seem to consider it more suggestion than requirement.

You know it is bad news for the company when they request pages and pages of redactions:

This recently happened with Amazon as well, and it turned out they were trying to hide the existence of a price-hiking algorithm that skimmed billions from
consumers. But it’s much worse when you’re redacting COPPA complaints.

“We’re very bullish and confident in our COPPA allegations. Meta is knowingly taking steps that harm children, and lying about it,” AG Bonta told TechCrunch in
an interview. “In the unredacted complaint we see that Meta knows that its social media platforms are used by millions of kids under 13, and they unlawfully
collect their personal info. It shows that common practice where Meta says one thing in its public-facing comments to Congress and other regulators, while
internally it says something else.”

Graph facts:
- Meta -MENTIONS-> Al-Aqsa Mosque
- Meta -MENTIONS-> WhatsApp
- Meta -MENTIONS-> Instagram Stories
- Meta -MENTIONS-> Palestine
- Meta -MENTIONS-> Colombia

Entities: Meta, TechCrunch, Congress, Meta turned a blind eye to kids on its platforms for years, unredacted lawsuit alleges, AG Bonta, price-hiking algorithm,
harm to children, lying about it, unlawful collection of children's personal information, COPPA, Amazon
</item 10>
=== END ITEM id=e2e17ed50c5bebd4 ===

=== ITEM id=d70e129a791b0e70 ===
Gold sentences and facts:
Sentence 0: But by the age of 16, she is public property – incapable of going outside without being mobbed, and her love of singing and dancing now a lucrative resource.
  Fact 0.0: By the age of 16, the person described is public property.
  Fact 0.1: By the age of 16, the person described is incapable of going outside without being mobbed.
  Fact 0.2: By the age of 16, the person described's love of singing and dancing is now a lucrative resource.
Sentence 1: "It was almost like this very strange, subtle clue to the media that they don't know everything that happened in that relationship, and I can have something really major and traumatic happen to me and they don't know about it," she told NPR.
  Fact 1.0: The speaker described it as almost a very strange, subtle clue to the media.
  Fact 1.1: The clue signals that the media does not know everything that happened in that relationship.
  Fact 1.2: The speaker says something really major and traumatic could have happened to her that the media does not know about.
  Fact 1.3: The speaker told NPR this.

Retrieved items:
<item 1>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
It’s posting selfies – either nude or clothed – on Instagram. Some people don’t understand the latter, she writes, “but I think if they’d been photographed by
other people thousands of times, prodded and posed for other people’s approval, they’d understand that I get a lot of joy from posing the way I feel sexy and
taking my own picture, doing whatever I want with it.”

Her ‘virginity’ was a prison

Britney was actively marketed as a virgin, a repulsive, archaic premise that made the 16-year-old’s inevitable sexuality into a kind of timebomb. In fact, she
writes happily, she first slept with her older brother’s best friend when she was 14. Yet the fixation on her “purity” took “the focus off me as a musician and
a performer”, she writes. “All some reporters could think of to ask me was whether or not my breasts were real (they were, actually) and whether or not my hymen
was intact.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Britney Spears’ memoir, virginity, sexuality, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir,
The Guardian, Instagram
</item 1>

<item 2>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Rock in Rio, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 2>

<item 3>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
The conservatorship stripped me of my womanhood, made me into a child.” It’s hard to explain, she says, “how quickly I could vacillate between being a little
girl and being a teenager and being a woman, because of the way they had robbed me of my freedom … They wanted me to be wild onstage, the way they told me to
be, and to be a robot the rest of the time.”

Music seems to have become an afterthought

Early on, Britney writes, she loved making music. She told the producer Max Martin that she wanted to sound more R&B than straight pop; she listened to Tainted
Love the night before recording Baby, One More Time and stayed up late so that she would sound as gravelly and “fried” as Marc Almond. She was a studio
perfectionist who would stay in the booth for hours, and the school setting and uniforms of the Baby video were her idea.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, Tainted Love, Baby, One More Time, Marc Almond, R&B, straight pop, Baby video, ‘I was being ritually tortured’: the most shocking
moments in Britney Spears’ memoir, Max Martin, The Guardian, music
</item 3>

<item 4>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
Her life under the conservatorship is a horror story

It feels fitting that The Woman in Me is being released a week before Halloween. No matter how much we already know about the 13 years that Britney lived under
a conservatorship, her recounting of its imposition, life under it and her inability to escape it is nothing short of a horror story. It is appalling, like
something out of a Victorian novel, not the very real experiences of a thirtysomething in the last decade.

When she fears that Federline is going to take away access to her kids, she locks herself in a bathroom with one of her sons. “Before I knew what was happening,
a Swat team in black suits burst through the bathroom door as if I’d hurt someone,” she writes, and she is subject to a forced hospital hold. Soon after, her
mother invites her to her beach house because “the cops are after you”. When she arrives, so does another Swat team.

Graph facts:
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Federline
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Federline
- Federline -PARTICIPATED_IN-> Marriage
- Federline -PARTICIPATED_IN-> divorce
- Federline -PARTICIPATED_IN-> filing for divorce

Entities: Federline, Britney Spears, conservatorship, forced hospital hold, Swat team, her mother, ‘I was being ritually tortured’: the most shocking moments in
Britney Spears’ memoir, Halloween, The Guardian, The Woman in Me
</item 4>

<item 5>
Graph query row:
article: ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir
publisher: The Guardian
article_entity: Justin Timberlake
relationship: MENTIONS
article_passage_source: ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
article_passage: ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
She has never really known freedom

As a child, Britney escaped her difficult home life – her alcoholic father often rowing with her mother – into the outdoors, which “gave me a sense of aliveness
and danger”, she writes. Performing also makes her feel invincible. But by the age of 16, she is public property – incapable of going outside without being
mobbed, and her love of singing and dancing now a lucrative resource.

By the time of her second album, she wants for so little: skinny-dipping with her dancers after playing Rock in Rio in January 2001 is one of her happiest
memories. “It was a taste of rebellion and freedom, but I was just having fun and being a 19-year-old,” she writes.
entity_passage_source: From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style,
2023-10-18T03:46:39+00:00)
entity_passage: From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships (The Independent - Life and Style, 2023-10-18T03:46:39+00:00)
Stay ahead of the trend in fashion and beyond with our free weekly Lifestyle Edit newsletter Stay ahead of the trend in fashion and beyond with our free weekly
Lifestyle Edit newsletter Please enter a valid email address Please enter a valid email address SIGN UP I would like to be emailed about offers, events and
updates from The Independent. Read our privacy notice Thanks for signing up to the

Lifestyle Edit email {{ #verifyErrors }} {{ message }} {{ /verifyErrors }} {{ ^verifyErrors }} Something went wrong. Please try again later {{ /verifyErrors }}

In her forthcoming memoir, The Woman in Me, Britney Spears is expected to reflect on some of the most pivotal moments in her life, including her relationships.

Over the years, the pop star’s love life has continued to make headlines. Most recently, her ex-husband Sam Asghari filed for divorce after only 14 months of
marriage.
</item 5>

<item 6>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
When she arrives, so does another Swat team. “I’m a five-foot four-inch pop singer who calls everyone sir and ma’am,” she writes. “They treated me like I was a
criminal or predator.”

The conservatorship clamps down on her life, ruled by her father, who she alleges was “an alcoholic, someone who’d declared bankruptcy, who’d failed in
business, who’d terrified me as a little girl”. She is medicated and surveilled. Her dates are required to agree to background checks and blood tests, made to
sign NDAs, and receive a full rundown of her sexual history before the first date. “The insanity of this system kept me from finding basic companionship, having
a fun night out or making new friends – let alone falling in love,” Britney writes. “It was just the worst thing that could possibly ever happen to my music, my
career and my sanity.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, Swat team, her father, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The
Guardian
</item 6>

<item 7>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
There is no love lost for her family

Britney writes about how she was always made to feel she was not enough as a kid; her father’s alleged alcoholism, neglect and brutal standards; how her mother
always made her feel less-than. She is unsparing about her younger sister, who she characterises as a brat who was spared the pain of Britney’s impoverished
childhood because of her success. “My mom and I had to witness the ugliness and the violence without believing that there was anywhere else to go.”

Even once Britney is the biggest pop star in the world, her parents still seem to have an undue say in the running of her career, including making her sit for
humiliating TV interviews – why exactly that is remains unanswered. Her father seizes control legally with the imposition of the conservatorship, telling her,
chillingly, “I am Britney Spears now,” and her family appear to live off her dime while she lives an extraordinarily restricted life.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, ‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 7>

<item 8>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
As the restrictions get tighter, Britney acts out – and ends up legally denied her freedom for 13 years, under a conservatorship controlled by her father.

Her body felt like public property

From day one, Britney’s body is made fair game. Between the videos for her debut single, Baby, One More Time and Sometimes, the media noted her changing
physique and concluded that the 16-year-old must have had breast implants – rather than simply going through puberty. She notices that talkshow hosts ask her
about her chest, while her boyfriend Justin Timberlake gets asked serious questions about his music. It’s one of many pressures that ultimately push her to
breaking point, and when she writes about shaving her head in a salon in 2007, she frames it as a “fuck you” to the impossible beauty standards she felt forced
to live up to: “I’d smiled politely while TV show hosts leered at my breasts, while American parents said I was destroying their children by wearing a crop
top.”

Graph facts:
- Justin Timberlake -PARTICIPATED_IN-> Britney Spears and Justin Timberlake breakup
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Justin Timberlake
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Justin Timberlake
- Justin Timberlake -PARTICIPATED_IN-> American Music Awards
- Justin Timberlake -PART_OF-> NSYNC

Entities: Justin Timberlake, Britney Spears, conservatorship, Baby, One More Time, Sometimes, impossible beauty standards, shaving her head, ‘I was being
ritually tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 8>

<item 9>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
At what point did I promise to stay 17 for the rest of my life?”

What she knows now, she writes, “is that every part of normal life had been stripped from me – going out in public without becoming a headline, making normal
mistakes as a new mother of two babies, feeling like I could trust the people around me. I had no freedom and yet also no security. At the time I was also
suffering, I now know, from severe postpartum depression.” It left her suicidal. “I’ll admit it, I felt that I couldn’t live if things didn’t get better.”

She knows she is trapped in a state of arrested development

It’s often said that famous people are frozen in time at the age they became famous, but Britney was never allowed to be a real child, nor a real adult. Her
2001 single I’m Not a Girl, Not Yet a Woman may be her most prescient recording; she is painfully conscious of this divide.

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, I’m Not a Girl, Not Yet a Woman, postpartum depression, arrested development, ‘I was being ritually tortured’: the most shocking
moments in Britney Spears’ memoir, The Guardian
</item 9>

<item 10>
‘I was being ritually tortured’: the most shocking moments in Britney Spears’ memoir (The Guardian, 2023-10-24T18:37:25+00:00)
After her split from Timberlake, she likens herself to Benjamin Button. “Somehow that year, in becoming more vulnerable I started to feel like a child again.”

The situation worsens amid postnatal depression. “It was as if some part of me became the baby,” she writes in an apologetic section about transferring her
frustrations on to her interior decorator. “One part of me was a very demanding grown woman yelling about white marble, while another part of me was suddenly
very childlike.”

When she is placed under the conservatorship, she not only regresses, but feels dehumanised by the constant scrutiny. “I became a robot. But not just a robot –
a sort of child-robot. The conservatorship stripped me of my womanhood, made me into a child.”

Graph facts:
- Britney Spears -MENTIONS-> ABC News
- From Justin Timberlake to Sam Asghari: A timeline of Britney Spears’ relationships -MENTIONS-> Britney Spears
- Britney Spears -MENTIONS-> Instagram Stories
- Charlie Sheen is 'single dad' to twins, follows John Travolta, Liam Neeson raising kids solo -MENTIONS-> Britney Spears
- Britney Spears and Kevin Federline: A complete timeline of their relationship -MENTIONS-> Britney Spears

Entities: Britney Spears, conservatorship, Britney Spears’ memoir, postnatal depression, Timberlake, Benjamin Button, interior decorator, ‘I was being ritually
tortured’: the most shocking moments in Britney Spears’ memoir, The Guardian
</item 10>
=== END ITEM id=d70e129a791b0e70 ===
